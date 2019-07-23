#include <iostream>
#include <stdint.h>
#include <string>

#include "murmuda3.h"

int main() {

    int num_keys = 1;
    int num_seeds = 10;
    int m = 1024;

    std::string* cpp_keys = new std::string("Hello");
    char const *keys = cpp_keys->c_str();
    int len = sizeof(keys);

    uint32_t* seeds = new uint32_t[num_seeds];
    for (int i = 0; i < num_seeds; i++) {
        seeds[i] = i;
    }

    uint32_t* out = new uint32_t[num_keys * num_seeds];

    MurmurHash3_batch(keys, len, num_keys, seeds, num_seeds, out);

    uint32_t* c_out = new uint32_t[1];
    for (int i = 0; i < num_keys; i++) {
        for (int j = 0; j < num_seeds; j++) {
            std::cout << out[j + i * num_seeds] % m << std::endl;
        }
    }
}
