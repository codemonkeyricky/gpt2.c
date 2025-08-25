
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
    int n_heads;
    int g_heads;
    int n_head_dim;
    int kv_heads;
    int n_layers;
    int seq_len;

    struct Derived d;
};

struct Layer {
    const __bf16 *input_layernorm;
    const __bf16 *post_attn_layernorm;
    const __bf16 *q_proj_w;
    const __bf16 *q_proj_b;
    const __bf16 *k_proj_w;
    const __bf16 *k_proj_b;
    const __bf16 *v_proj_w;
    const __bf16 *v_proj_b;
    const __bf16 *o_proj_w;
    const __bf16 *mlp_gate_proj;
    const __bf16 *mlp_up_proj;
    const __bf16 *mlp_down_proj;
};

struct Mmapping {
    const __bf16 *embeddings;
    struct Layer *layers;
};

typedef struct {
    __bf16 *cache;
} Head;

struct RLayer {
    Head *key;
    Head *value;
};

struct Runtime {
    __bf16 *q;
    __bf16 *k;
    __bf16 *v;
    __bf16 *h1;
    __bf16 *h2;
    __bf16 *h3;
    struct RLayer *layers;
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

    fd = open("layer_0_o_proj_w.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->o_proj_w = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_post_attention_layernorm.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->post_attn_layernorm = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_mlp_down_proj.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->mlp_down_proj = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_mlp_up_proj.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->mlp_up_proj = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("layer_0_mlp_gate_proj.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    l0->mlp_gate_proj = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
}

void config_init(struct Config *config) {
    config->vocab_size = 152064;
    config->n_embed = 2048;
    config->n_heads = 16;
    config->kv_heads = 2;
    config->n_layers = 36;
    config->seq_len = 1024; /* TODO: */
    config->n_head_dim = 128;

    rope_init(config);
}

void layernorm(__bf16 *out, const __bf16 *in, const __bf16 *weight, struct Transformer *x) {

    struct Config *c = &x->config;
    struct Mmapping *m = &x->mmapping;

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

void mul(__bf16 *out, const __bf16 *a, const __bf16 *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (__bf16)((float)a[i] * (float)b[i]);
    }
}

void add(__bf16 *out, const __bf16 *a, const __bf16 *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (__bf16)((float)a[i] + (float)b[i]);
    }
}

void rotate_half(__bf16 *out, const __bf16 *x, int D) {
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        out[i] = -x[i + half]; // Negate the second half and put it in the first half
        out[i + half] = x[i];  // Copy the first half to the second half
    }
}

void rotary_positional_embedding(__bf16 *emb, __bf16 *cos, __bf16 *sin, const struct Transformer *x) {
    __bf16 *in = emb;
    __bf16 *out = x->runtime.h1;
    __bf16 *out2 = x->runtime.h2;
    __bf16 *out3 = x->runtime.h3;

    int n = x->config.n_head_dim;

    /* a = rotate_half(q) * sin */
    rotate_half(out, in, n);
    mul(out2, out, sin, n);

    /* b = q * cos */
    mul(out, in, cos, n);

    /* a + b */
    add(out3, out, out2, n);

    memcpy(emb, out3, n * sizeof(__bf16));
}

void self_attention(__bf16 *__restrict xout, __bf16 *__restrict x, const struct Transformer *xfmr, const int layer,
                    const int pos, __bf16 *sin, __bf16 *cos) {

    const struct Config *p = &xfmr->config;
    const struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;

    const __bf16 *qw = m->layers[layer].q_proj_w; // weight for the query projection
    const __bf16 *qb = m->layers[layer].q_proj_b; // bias for the query projection

    const __bf16 *kw = m->layers[layer].k_proj_w; // weight for the key projection
    const __bf16 *kb = m->layers[layer].k_proj_b; // bias for the key projection

    const __bf16 *vw = m->layers[layer].v_proj_w; // weight for the value projection
    const __bf16 *vb = m->layers[layer].v_proj_b; // bias for the value projection

    const __bf16 *ow = m->layers[layer].o_proj_w; // weight for the output projection

    /* attention weight and bias */
    matmul_bias(r->q, x, qw, qb, p->n_embed, p->n_embed);
    matmul_bias(r->k, x, kw, kb, p->n_embed, 256);
    matmul_bias(r->v, x, vw, vb, p->n_embed, 256);

    rotary_positional_embedding(r->q, cos, sin, xfmr);
    rotary_positional_embedding(r->k, cos, sin, xfmr);

    /* insert to kv cache */
    int n_heads = 16, kv_heads = 2;
    int index = n_heads / kv_heads;
    size_t hs = 128;
    for (size_t h = 0; h < 2; h++) {
        memcpy(r->layers[layer].key[h].cache + pos * hs, r->k + h * hs, hs * sizeof(__bf16));
        memcpy(r->layers[layer].value[h].cache + pos * hs, r->v + h * hs, hs * sizeof(__bf16));
    }

    /* Calculate attention score */
    __bf16 att[pos + 1] = {};
    __bf16 *y = xout;
    memset(y, 0, p->n_embed * sizeof(__bf16)); // clear output buffer
    for (int h = 0; h < p->n_heads; h++) {

        /* find the query head */
        const __bf16 *qq = r->q + h * hs; // (1, hs)
        for (int t = 0; t <= pos; t++) {
            __bf16 *kk = r->layers[layer].key[h / index].cache + t * hs; // (T, hs)
            float score = 0.0f;
            for (int i = 0; i < hs; i++) {
                score += (float)qq[i] * (float)kk[i];
            }
            att[t] = (__bf16)score;
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
            __bf16 *vv = r->layers[layer].value[h / index].cache;
            __bf16 *yy = y + h * hs; // (1, hs)
            for (int t = 0; t <= pos; t++) {
                /* find v for the current head */
                yy[i] += att[t] * vv[t * hs + i];
            }
        }
    }

    /* TODO */
    memcpy(x, y, p->n_embed * sizeof(__bf16));
    matmul(y, x, ow, p->n_embed, p->n_embed);

    volatile int dummy = 0;
}

void runtime_init(struct Transformer *xfmr) {

    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;

    r->q = (__bf16 *)malloc(sizeof(__bf16) * c->n_embed);
    r->k = (__bf16 *)malloc(sizeof(__bf16) * 256);
    r->v = (__bf16 *)malloc(sizeof(__bf16) * 256);

    r->h1 = (__bf16 *)malloc(sizeof(__bf16) * 256);
    r->h2 = (__bf16 *)malloc(sizeof(__bf16) * 256);
    r->h3 = (__bf16 *)malloc(sizeof(__bf16) * 256);

    r->layers = (struct RLayer *)calloc(sizeof(struct RLayer), c->n_layers);

    for (size_t i = 0; i < c->n_layers; i++) {
        r->layers[i].key = (Head *)calloc(sizeof(Head), 2);
        r->layers[i].value = (Head *)calloc(sizeof(Head), 2);
        for (size_t j = 0; j < 2; ++j) {
            /* TODO: parameterized 2 and 128, relating to group heads  */
            r->layers[i].key[j].cache = (__bf16 *)calloc(sizeof(__bf16), c->seq_len * 128);
            r->layers[i].value[j].cache = (__bf16 *)calloc(sizeof(__bf16), c->seq_len * 128);
        }
    }
}

void silu_array(__bf16 *output, const __bf16 *input, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] / (1.0f + expf(-input[i]));
    }
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

    __bf16 embeddings[c->n_embed] = {}, embeddings2[c->n_embed] = {}, skip[c->n_embed] = {},
           embeddings3[c->n_embed] = {};

    int token = 151644;
    memcpy(embeddings, m->embeddings + token * c->n_embed, c->n_embed * sizeof(__bf16));

    // tensor([[151644,    872,    198,   1944, 151645,    198, 151644,  77091,    198]])

    __bf16 cos[128] = {}, sin[128] = {};

    rope_forward(&c->d.rope, 1, cos, sin);

    {
        /* save skip */
        memcpy(skip, embeddings, c->n_embed * sizeof(__bf16));

        layernorm(embeddings2, embeddings, m->layers[0].input_layernorm, x);
        self_attention(embeddings, embeddings2, x, 0, 0, sin, cos);

        /* residual */
        add(embeddings2, embeddings, skip, c->n_embed);
    }

    {
        /* save skip */
        memcpy(skip, embeddings2, c->n_embed * sizeof(__bf16));

        layernorm(embeddings, embeddings2, m->layers[0].post_attn_layernorm, x);

        __bf16 embeddings2[11008] = {}, embeddings3[11008] = {}, embeddings4[11008] = {};

        /* up proj */
        matmul(embeddings2, embeddings, m->layers[0].mlp_up_proj, c->n_embed, 11008);

        /* gate proj */
        matmul(embeddings3, embeddings, m->layers[0].mlp_gate_proj, c->n_embed, 11008);
        silu_array(embeddings2, embeddings3, 11008);

#if 0
        mul(embeddings3, embeddings, embeddings2, c->n_embed);

        /* down proj */
        matmul(embeddings, embeddings3, m->layers[0].mlp_down_proj, c->n_embed, c->n_embed);
#endif
    }

    volatile int dummy = 0;
}