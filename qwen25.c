
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

struct Config {
    int vocab_size;
    int n_embed;
};

struct Mmapping {
    const __bf16 *embeddings;
};

struct RotaryPosEmb {
    float inv_freq[64];
};

void rope_init(struct Config *c, struct RotaryPosEmb *rope) {
    const float t = 1000000.0;
    for (int i = 0; i < 64; i++) {
        rope->inv_freq[i] = 1.0f / powf(t, (2.0f * i) / c->n_embed);
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

int main() {

    struct Mmapping mmapping = {};
    struct Config config = {
        .vocab_size = 152064,
        .n_embed = 2048,
    };

    int fd = open("embeddings.bin", O_RDONLY);
    assert(fd > -1);
    int file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    mmapping.embeddings = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    struct Config *c = &config;

    struct RotaryPosEmb rope = {};
    rope_init(c, &rope);

    __bf16 embeddings[c->n_embed] = {};

    int token = 151644;
    memcpy(embeddings, mmapping.embeddings + token * c->n_embed, c->n_embed * sizeof(__bf16));

    // tensor([[151644,    872,    198,   1944, 151645,    198, 151644,  77091,    198]])

    volatile int dummy = 0;
}