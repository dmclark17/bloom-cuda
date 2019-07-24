#include <algorithm>
#include <iostream>

#include "bloom.h"
#include "murmuda3.h"

__global__
void cuda_add(uint32_t* cuda_bit_vector, int num_bits, uint32_t* cuda_seeds,
              int num_seeds, const void* cuda_key, int len) {
    // For now lets pretend this is just called for one key
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Allocate memory on device from kernel for output of hash. This is init
    // by the decorator
    extern __shared__ uint32_t out[];
    uint32_t bit_index;

    // Hash them in parallel
    for (int k = index; k < num_seeds; k+= stride) {
        _Murmur3_helper(cuda_key, len, cuda_seeds[k], &(out[k]));

        // Use cuda atomic functions to guarentee it is flipped
        bit_index = out[k] % num_bits;
        atomicOr(&(cuda_bit_vector[bit_index / 32]),
                 (uint32_t) 1 << (bit_index % 32));
    }
}

__global__
void cuda_test(uint32_t* cuda_bit_vector, int num_bits, uint32_t* cuda_seeds,
               int num_seeds, const void* cuda_key, int len, bool * bool_out) {
    // For now lets pretend this is just called for one key
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Allocate memory on device from kernel for output of hash. This is init
    // by the decorator
    extern __shared__ bool test_vals[];
    uint32_t out, bit_index;

    // Hash them in parallel
    for (int k = index; k < num_seeds; k+= stride) {
        _Murmur3_helper(cuda_key, len, cuda_seeds[k], &out);
        bit_index = out % num_bits;
        test_vals[k] = (cuda_bit_vector[bit_index / 32] & (1 << (bit_index % 32)));
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        *bool_out = true;
        for (int i = 0; i < num_seeds; i++) {
            if (!test_vals[i]) {
                *bool_out = false;
                break;
            }
        }
    }
}


BloomFilter::BloomFilter(int n_bits, int n_seeds) {
    num_bits = n_bits;

    num_int = (num_bits + (sizeof(uint32_t) - 1)) / sizeof(uint32_t);
    bit_vector =  new uint32_t[num_int];
    std::fill(bit_vector, bit_vector+num_int, 0);

    // Allocate the bit vector on the device
    cudaMalloc(&cuda_bit_vector, num_int * sizeof(bit_vector[0]));
    cudaMemcpy(cuda_bit_vector, bit_vector, num_int * sizeof(bit_vector[0]),
               cudaMemcpyHostToDevice);


    num_seeds = n_seeds;
    seeds = new uint32_t[num_seeds];
    for (int i = 0; i < num_seeds; i++) {
        seeds[i] = i;
    }

    cudaMalloc(&cuda_seeds, num_seeds * sizeof(uint32_t));
    cudaMemcpy(cuda_seeds, seeds, num_seeds * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
}


void BloomFilter::add(const void * key, int len) {
    void * cuda_key;
    cudaMalloc(&cuda_key, len);
    cudaMemcpy(cuda_key, key, len, cudaMemcpyHostToDevice);

    int blockSize = num_seeds;
    int numBlocks = 1;

    cuda_add<<<numBlocks,
               blockSize,
               num_seeds * sizeof(uint32_t)>>>(cuda_bit_vector, num_bits,
                                               cuda_seeds, num_seeds,
                                               cuda_key, len);

    cudaDeviceSynchronize();

    cudaFree(cuda_key);
}

void BloomFilter::sync() {
    cudaMemcpy(bit_vector, cuda_bit_vector, num_int * sizeof(bit_vector[0]),
               cudaMemcpyDeviceToHost);
}

bool BloomFilter::test(const void * key, int len) {
    bool result;
    bool * cuda_result;
    cudaMalloc(&cuda_result, sizeof(bool));

    void * cuda_key;
    cudaMalloc(&cuda_key, len);
    cudaMemcpy(cuda_key, key, len, cudaMemcpyHostToDevice);

    int blockSize = num_seeds;
    int numBlocks = 1;

    cuda_test<<<numBlocks,
                blockSize,
                num_seeds * sizeof(uint32_t)>>>(cuda_bit_vector, num_bits,
                                                cuda_seeds, num_seeds,
                                                cuda_key, len, cuda_result);

    cudaDeviceSynchronize();

    cudaMemcpy(&result, cuda_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(cuda_key);
    cudaFree(cuda_result);

    return result;
}
