#include <iostream>
#include <stdint.h>
#include <string>
#include <cmath>

#include "bloom.h"


int main() {
    int num_bits = 4096;
    int num_keys = 100;
    int num_test_keys = 5000;

    int num_seeds = (int) ( (float)num_bits / num_keys * std::log(2));
    BloomFilter* bf = new BloomFilter(num_bits, num_seeds);
    std::cout << "Number of hash functions " << num_seeds << std::endl;

    int32_t* keys = new int32_t[num_keys];
    int len = sizeof(keys[0]);
    for (int i = 0; i < num_keys; i++) {
        keys[i] = i;
    }

    for (int i = 0; i < num_keys; i++) {
        bf->add(&keys[i], len);
    }

    bf->sync();

    for (int i = 0; i < num_keys; i++) {
        if (!bf->test(&keys[i], len)) {
            std::cout << "Error for key" << keys[i] << std::endl;
        }
    }

    int32_t* test_keys = new int32_t[num_test_keys];
    for (int i = 0; i < num_test_keys; i++) {
        test_keys[i] = i + num_keys;
    }

    int false_positives = 0;
    for (int i = 0; i < num_test_keys; i++) {
        if (bf->test(&test_keys[i], len)) {
            false_positives++;
        }
    }
    float rate = ((float) false_positives) / num_test_keys;

    float expo = (((float) num_seeds) * num_keys) / num_bits;
    float base = 1 - std::exp(-1 * expo);
    float expected = std::pow(base, num_seeds);
    std::cout << "False positive: " << rate << std::endl;
    std::cout << "Expected: " << expected << std::endl;
}
