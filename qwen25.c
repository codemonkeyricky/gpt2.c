
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
};

struct Mmapping {
    const __bf16 *embeddings;
    struct Layer *layers;
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

int main() {

    struct Mmapping mmapping = {};
    struct Config config = {};
    struct RotaryPosEmb rope = {};

    config_init(&config);
    mmap_init(&config, &mmapping);

    struct Config *c = &config;
    struct Mmapping *m = &mmapping;

    __bf16 embeddings[c->n_embed] = {}, embeddings2[c->n_embed] = {};

    int token = 151644;
    memcpy(embeddings, mmapping.embeddings + token * c->n_embed, c->n_embed * sizeof(__bf16));

    // tensor([[151644,    872,    198,   1944, 151645,    198, 151644,  77091,    198]])

    __bf16 cos[64] = {}, sin[64] = {};

    rope_forward(&rope, 1, cos, sin);

    input_layernorm(embeddings2, embeddings, c, m);

    volatile int dummy = 0;
}