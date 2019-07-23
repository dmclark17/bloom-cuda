#ifndef _BLOOM_CUDA_H_
#define _BLOOM_CUDA_H_

#include <stdint.h>

class BloomFilter {
        int num_bits; // Length of bit vector
        int num_int; // Length of array
        uint16_t* bit_vector;
        int num_seeds; // number of hash functions
        uint32_t * seeds;

        uint16_t* cuda_bit_vector;
        uint32_t* cuda_seeds;
    public:
        BloomFilter() {}
        BloomFilter(int n_bits, int n_seeds);
        void add(const void * key, int len);
        int batch_add(const void * key, int len, int num);
        void sync();
        bool test(const void * key, int len);
        int batch_test(const void * key, int len, int num);
};


#endif // _BLOOM_CUDA_H_
