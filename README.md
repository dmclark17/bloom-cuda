# bloom-cuda

bloom filter implemented in CUDA for fun. Able to add/test batches of keys in parallel

## TODO

- Currently does not hash all the seeds in parallel. Need to use hashing function which batches both on keys and seeds
