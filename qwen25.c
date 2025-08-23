
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

struct RotaryPosEmb {
    float inv_freq[64];
};

struct Derived {
    struct RotaryPosEmb rope;
};

struct Config {
    int vocab_size;
    int n_embed;
    struct Derived d;
};

struct Layer {
    const __bf16 *input_layernorm;
    const __bf16 *q_proj_w;
    const __bf16 *q_proj_b;
    const __bf16 *k_proj_w;
    const __bf16 *k_proj_b;
    const __bf16 *v_proj_w;
    const __bf16 *v_proj_b;
};

struct Mmapping {
    const __bf16 *embeddings;
    struct Layer *layers;
};

struct Runtime {
    __bf16 *q;
    __bf16 *k;
    __bf16 *v;
};

struct Transformer {
    struct Config config;
    struct Mmapping mmapping;
    struct Runtime runtime;
};

void rope_init(struct Config *c) {

    // t = config.rope_theta
    // r = torch.arange(0, d, 2)

    // self.inv_freq = 1.0 / (t ** (r / d)).float()

    const float t = 1000000.0;
    const int d = 128;

    // Calculate inverse frequencies
    for (int i = 0; i < d / 2; i++) {
        float r = (float)(i * 2); // r = 0, 2, 4, ..., d-2
        float exponent = r / (float)d;
        c->d.rope.inv_freq[i] = 2.0f / powf(t, exponent);
    }
}

// {
//   "architectures": [
//     "Qwen2ForCausalLM"
//   ],
//   "attention_dropout": 0.0,
//   "bos_token_id": 151643,
//   "eos_token_id": 151645,
//   "hidden_act": "silu",
//   "hidden_size": 3584,
//   "initializer_range": 0.02,
//   "intermediate_size": 18944,
//   "max_position_embeddings": 32768,
//   "max_window_layers": 28,
//   "model_type": "qwen2",
//   "num_attention_heads": 28,
//   "num_hidden_layers": 28,
//   "num_key_value_heads": 4,
//   "rms_norm_eps": 1e-06,
//   "rope_theta": 1000000.0,
//   "sliding_window": 131072,
//   "tie_word_embeddings": false,
//   "torch_dtype": "bfloat16",
//   "transformers_version": "4.43.1",
//   "use_cache": true,
//   "use_sliding_window": false,
//   "vocab_size": 152064
// }

void rope_forward(struct RotaryPosEmb *rope, int seq_len, __bf16 *cos, __bf16 *sin) {
    // position_ids = torch.arange(seq_len, dtype=torch.float).to(device)
    // freqs = position_ids[:, None] * self.inv_freq[None, :]

    for (int p = 0; p < seq_len; p++) {
        for (int f = 0; f < 64; f++) {
            float freq = (float)p * rope->inv_freq[f];
            cos[p * 64 + f] = cosf(freq);
            sin[p * 64 + f] = sinf(freq);
        }
    }
}

void mmap_init(struct Config *config, struct Mmapping *mmapping) {
    int fd = open("embeddings.bin", O_RDONLY);
    assert(fd > -1);
    int file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    mmapping->embeddings = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    mmapping->layers = (struct Layer *)malloc(sizeof(struct Layer) * 28);

    struct Layer *l0 = &mmapping->layers[0];

    fd = open("layer_0_input_layernorm.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->input_layernorm = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_q_proj_w.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->q_proj_w = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_k_proj_w.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->k_proj_w = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_v_proj_w.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->v_proj_w = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_q_proj_b.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->q_proj_b = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_k_proj_b.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->k_proj_b = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_v_proj_b.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->v_proj_b = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
}

void config_init(struct Config *config) {
    config->vocab_size = 152064;
    config->n_embed = 2048;

    rope_init(config);
}

void input_layernorm(__bf16 *out, const __bf16 *in, struct Config *c, struct Mmapping *m) {
    const __bf16 *weight = m->layers[0].input_layernorm;
    // const __bf16 *bias = m->layers[0].input_layernorm_bias;

    // mean = x.mean(-1, keepdim=True)
    // variance = x.var(-1, keepdim=True, unbiased=False)
    // x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
    // return x * self.weight + self.bias

    float mean = 0.0f;
    for (int i = 0; i < c->n_embed; i++) {
        mean += (float)in[i];
    }
    mean /= (float)c->n_embed;

    float variance = 0.0f;
    for (int i = 0; i < c->n_embed; i++) {
        float diff = (float)in[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)c->n_embed;

    // def forward(self, x):
    //     input_dtype = x.dtype
    //     x = x.to(torch.float32)
    //     variance = x.pow(2).mean(-1, keepdim=True)
    //     x = x * torch.rsqrt(variance + self.variance_epsilon)
    //     return self.weight * x.to(input_dtype)

    float denom = 1.0f / sqrtf(variance + 1e-6f);

    for (int i = 0; i < c->n_embed; i++) {
        out[i] = (__bf16)(in[i] * denom * (float)weight[i]);
    }
}

void matmul(__bf16 *__restrict xout_in, const __bf16 *__restrict x_in, const __bf16 *__restrict w_in, int n, int d) {
    __bf16 *xout = xout_in;
    const __bf16 *x = x_in;
    const __bf16 *w = w_in;
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += (float)w[i * n + j] * (float)x[j];
        }
        xout[i] = (__bf16)sum;
    }
}

void matmul_bias(__bf16 *out, const __bf16 *x, const __bf16 *w, const __bf16 *b, int n, int d) {
    matmul(out, x, w, n, d);
    for (size_t i = 0; i < d; i++) {
        out[i] += b[i];
    }
}

void self_attention(__bf16 *__restrict xout, __bf16 *__restrict x, const struct Transformer *xfmr, const int layer,
                    const int pos) {

    const struct Config *p = &xfmr->config;
    const struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;

    const __bf16 *qw = m->layers[layer].q_proj_w; // weight for the query projection
    const __bf16 *qb = m->layers[layer].q_proj_b; // bias for the query projection

    const __bf16 *kw = m->layers[layer].k_proj_w; // weight for the key projection
    const __bf16 *kb = m->layers[layer].k_proj_b; // bias for the key projection

    const __bf16 *vw = m->layers[layer].v_proj_w; // weight for the value projection
    const __bf16 *vb = m->layers[layer].v_proj_b; // bias for the value projection

    /* attention weight and bias */
    matmul_bias(r->q, x, qw, qb, p->n_embed, p->n_embed);
    matmul_bias(r->k, x, kw, kb, p->n_embed, 256);
    matmul_bias(r->v, x, vw, vb, p->n_embed, 256);

    volatile int dummy = 0;

#if 0
    /* split attention into q, k, v */
    const float *q = attn;
    const float *k = attn + p->dim;     // key
    const float *v = attn + p->dim * 2; // value

    /* Append current key/value to the cache */
    size_t hs = p->dim / p->n_heads;
    for (size_t h = 0; h < p->n_heads; h++) {
        memcpy(s->layers[layer].key[h].cache + pos * hs, k + h * hs, hs * sizeof(float));
        memcpy(s->layers[layer].value[h].cache + pos * hs, v + h * hs, hs * sizeof(float));
    }

    float *y = xout;
    memset(y, 0, p->dim * sizeof(float)); // clear output buffer

    /* Calculate attention score */
    float att[pos + 1] = {};
    for (int h = 0; h < p->n_heads; h++) {

        /* find the query head */
        const float *qq = q + h * hs; // (1, hs)
        for (int t = 0; t <= pos; t++) {
            float *kk = s->layers[layer].key[h].cache + t * hs; // (T, hs)
            float score = 0.0f;
            for (int i = 0; i < hs; i++) {
                score += qq[i] * kk[i];
            }
            att[t] = score;
        }

        for (int t = 0; t <= pos; t++) {
            att[t] /= sqrtf(hs);
        }

        /* soft max */

        float max_att = att[0];
        for (int t = 1; t <= pos; t++) {
            if (att[t] > max_att)
                max_att = att[t];
        }
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            att[t] = expf(att[t] - max_att);
            sum_exp += att[t];
        }
        for (int t = 0; t <= pos; t++) {
            att[t] /= sum_exp;
        }

        /* y = att @ v // (1, T) x (T, hs) -> (1, hs) */
        for (int i = 0; i < hs; i++) {
            float *vv = s->layers[layer].value[h].cache;
            float *yy = y + h * hs; // (1, hs)
            for (int t = 0; t <= pos; t++) {
                /* find v for the current head */
                yy[i] += att[t] * vv[t * hs + i];
            }
        }
    }

    memcpy(x, y, p->dim * sizeof(float));

    ww = w->h[layer].att.c_attn_proj_w; // weight for the projection
    bb = w->h[layer].att.c_attn_proj_b; // bias for the projection
    matmul_bias(y, x, ww, bb, p->dim, p->dim);
#endif
}

void runtime_init(struct Transformer *xfmr) {

    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;

    r->q = (__bf16 *)malloc(sizeof(__bf16) * c->n_embed);
    r->k = (__bf16 *)malloc(sizeof(__bf16) * 256);
    r->v = (__bf16 *)malloc(sizeof(__bf16) * 256);
}

int main() {

    // struct Mmapping mmapping = {};
    // struct Config config = {};
    // struct RotaryPosEmb rope = {};
    struct Transformer xfmr = {};

    struct Config *c = &xfmr.config;
    struct Mmapping *m = &xfmr.mmapping;
    struct Transformer *x = &xfmr;

    config_init(c);
    mmap_init(c, m);
    runtime_init(x);

    __bf16 embeddings[c->n_embed] = {}, embeddings2[c->n_embed] = {};

    int token = 151644;
    memcpy(embeddings, m->embeddings + token * c->n_embed, c->n_embed * sizeof(__bf16));

    // tensor([[151644,    872,    198,   1944, 151645,    198, 151644,  77091,    198]])

    __bf16 cos[64] = {}, sin[64] = {};

    rope_forward(&c->d.rope, 1, cos, sin);

    input_layernorm(embeddings2, embeddings, c, m);
    self_attention(embeddings, embeddings2, x, 0, 0);

    volatile int dummy = 0;
}