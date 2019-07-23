#include <algorithm>
#include <iostream>

#include "bloom.h"
#include "murmuda3.h"


BloomFilter::BloomFilter(int n_bits, int n_seeds) {
    num_bits = n_bits;

    int num_int = (num_bits + (sizeof(uint16_t) - 1)) / sizeof(uint16_t);
    bit_vector =  new uint16_t[num_int];
    std::fill(bit_vector, bit_vector+num_int, 0);

    num_seeds = n_seeds;
    seeds = new uint32_t[num_seeds];
    for (int i = 0; i < num_seeds; i++) {
        seeds[i] = i;
    }
}


void BloomFilter::add(const void * key, int len) {
    // Hash the key
    int num_keys = 1;
    uint32_t* out = new uint32_t[num_keys * num_seeds];

    MurmurHash3_batch(key, len, num_keys, seeds, num_seeds, out);

    // Add to bit_vector
    uint32_t index;
    for (int i = 0; i < num_seeds; i++) {
        index = out[i] % num_bits;
        // std::cout << index << std::endl;
        bit_vector[index / 16] |= 1 << (index % 16);
    }
}

bool BloomFilter::test(const void * key, int len) {
    // Hash the key
    int num_keys = 1;
    uint32_t* out = new uint32_t[num_keys * num_seeds];

    MurmurHash3_batch(key, len, num_keys, seeds, num_seeds, out);

    // Test the bit_vector
    uint32_t index;
    bool result = true;
    for (int i = 0; i < num_seeds; i++) {
        index = out[i] % num_bits;
        bool temp = (bit_vector[index / 16] & (1 << (index % 16)));
        result = result && temp;
        // std::cout << index << " " << temp << " " << result << std::endl;
    }
    return result;
}
