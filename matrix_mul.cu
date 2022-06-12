#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "util.h"

/**
This is ASDF billion saxpys per second for N = FDSA

It has `n` and `a` as parameters,
and there is a branch on the thread index being < n.
*/

#define M (uint64_t)20'000
#define N (uint64_t)20'000
#define K (uint64_t)20'000

#define N_A M * N
#define N_B N * K
#define N_C M * K

#define TOTAL_SIZE_A N_A * sizeof(float)
#define TOTAL_SIZE_B N_B * sizeof(float)
#define TOTAL_SIZE_C N_C * sizeof(float)

#define A_VAL 2.0f
#define B_VAL 5.0f


__global__
void matrix_mul(float *d_a, float *d_b, float *d_c) {
  // uint64_t tIdx = threadIdx.x + (blockDim.x * blockIdx.x);
  // if (tIdx < n) {
  //   d_z[tIdx] = (a * d_x[tIdx]) + d_y[tIdx];
  // }
}

int main(void) {
  printf("Matrix Mul Version 1\n");

  printf("Total size of A: %lld MB\n", TOTAL_SIZE_A / (1024 * 1024));
  printf("Total size of B: %lld MB\n", TOTAL_SIZE_B / (1024 * 1024));
  printf("Total size of C: %lld MB\n", TOTAL_SIZE_C / (1024 * 1024));

  cudaDeviceProp props;
  if (getDeviceProps(&props) != 0) {
    printf("Unable to get device props\n");
    return 1;
  }

  cudaError e;
  float *a, *b, *c;       // Host matrices
  float *d_a, *d_b, *d_c; // Device matrices

  // Allocate space on Host
  a = (float*)malloc(TOTAL_SIZE_A);
  b = (float*)malloc(TOTAL_SIZE_B);
  c = (float*)malloc(TOTAL_SIZE_C);

  // Allocate space on Device
  if ((e = cudaMalloc(&d_a, TOTAL_SIZE_A)) != CUDA_SUCC) {
    printf("Failed to allocate device A: %d\n", e);
    return 1;
  }
  if ((e = cudaMalloc(&d_b, TOTAL_SIZE_B)) != CUDA_SUCC) {
    printf("Failed to allocate device B: %d\n", e);
    return 1;
  }
  if ((e = cudaMalloc(&d_c, TOTAL_SIZE_C)) != CUDA_SUCC) {
    printf("Failed to allocate device C: %d\n", e);
    return 1;
  }

  // Set the values on Host for A and B
  for (uint64_t i = 0; i < N_A; i++) {
    a[i] = A_VAL;
  }
  for (uint64_t i = 0; i < N_B; i++) {
    b[i] = B_VAL;
  }

  // Copy values for A and B over to Device (C is for the result)
  if ((e = cudaMemcpy(d_a, a, TOTAL_SIZE_A, H_TO_D)) != CUDA_SUCC) {
    printf("Failed to copy A to device: %d\n", e);
    return 1;
  }
  if ((e = cudaMemcpy(d_b, b, TOTAL_SIZE_B, H_TO_D)) != CUDA_SUCC) {
    printf("Failed to copy B to device: %d\n", e);
    return 1;
  }

  // TODO @nocommit Continue here
  uint64_t l2Size = props.l2CacheSize;
  uint64_t maxSMPerBlock = props.sharedMemPerBlock;
  uint64_t smPerMP = props.sharedMemPerMultiprocessor;
  uint64_t mpCount = props.multiProcessorCount;
  uint64_t maxTotalSM = smPerMP * mpCount;
  uint64_t maxValsPerBlock = maxSMPerBlock / sizeof(float);
  uint64_t maxValsPerBlockPerMatrix = maxValsPerBlock / 2;
  uint64_t sidelenPerBlockPerMatrix = (uint64_t)sqrt(maxValsPerBlockPerMatrix);
  uint64_t valsPerBlockPerMatrix = sidelenPerBlockPerMatrix * sidelenPerBlockPerMatrix;
  uint64_t smPerBlockPerMatrix = valsPerBlockPerMatrix * sizeof(float);
  uint64_t nBlocks = maxTotalSM / smPerBlock;
  uint64_t nVals = nBlocks * valsPerBlock;
  uint64_t totalSM = smPerBlock * nBlocks;

  printf("L2 cache size: %lld\n", l2Size);
  printf("Max Shared Memory per Block: %lld\n", maxSMPerBlock);
  printf("Shared Memory per Multiprocessor: %lld\n", smPerMP);
  printf("N Multiprocessors: %lld\n", mpCount);
  printf("Max total shared memory: %lld\n", maxTotalSM);
  printf("Max vals per block: %lld\n", maxValsPerBlock);
  printf("Sidelen per block: %lld\n", sidelenPerBlock);
  printf("Vals per block: %lld\n", valsPerBlock);
  printf("Shared Memory per block: %lld\n", smPerBlock);
  printf("N blocks: %lld\n", nBlocks);
  printf("N vals across all blocks: %lld\n", nVals);
  printf("Total shared memory: %lld\n", totalSM);




  // If shared memory per block is 49152
  // And the size of a float is 4
  // Then we can handle s*s*4 <= 49152
  // block side length s = floor(sqrt(smPerBlock / sizeof(float)))
  // = floor(sqrt(49152 / 4)) = 110
  // So then dim.x = 110, dim.y = 110, dim.z = 1


  return -1;

  // // Block size
  // int warpSize = props.warpSize;
  // uint64_t maxBlockSizeX = props.maxThreadsDim[0];
  // int blockSize = warpSize * (maxBlockSizeX / warpSize);

  // // Grid size
  // uint64_t maxGridSizeX = props.maxGridSize[0];
  // uint64_t nBlocks = N / blockSize;
  // int nExtraThreads = N % blockSize;
  // if (nExtraThreads > 0) {
  //   ++nBlocks;
  // }

  // printf("N: %lld\n", (uint64_t)N);
  // printf("Warp size: %d\n", warpSize);
  // printf("Max block size X: %lld\n", maxBlockSizeX);
  // printf("Block size: %d\n", blockSize);
  // printf("Max grid size X: %lld\n", maxGridSizeX);
  // printf("N blocks: %lld\n", nBlocks);
  // printf("N extra threads (last block): %d\n", nExtraThreads);

  // if (nBlocks > maxGridSizeX) {
  //   printf("nBlocks > maxGridSizeX (%lld > %lld)\n", nBlocks, maxGridSizeX);
  //   return 1;
  // }

  // printf("A: %f\n", A);

  // saxpy<<<nBlocks, blockSize>>>(N, A, d_x, d_y, d_z);

  // // Copy Z back to Host
  // if ((e = cudaMemcpy(z, d_z, TOTAL_SIZE, D_TO_H)) != CUDA_SUCC) {
  //   printf("Failed to copy Z to host: %d\n", e);
  //   return 1;
  // }

  // // Free up the Device memory
  // if ((e = cudaFree(d_x)) != CUDA_SUCC) {
  //   printf("Failed to free device X: %d", e);
  //   return 1;
  // }
  // if ((e = cudaFree(d_y)) != CUDA_SUCC) {
  //   printf("Failed to free device Y: %d", e);
  //   return 1;
  // }
  // if ((e = cudaFree(d_z)) != CUDA_SUCC) {
  //   printf("Failed to free device Z: %d", e);
  //   return 1;
  // }

  // // Free up the host memory
  // free(x);
  // free(y);

  // // Do some stuff with the results
  // double sum = 0.0;
  // for (int i = 0; i < N; i++) {
  //   sum += z[i];
  // }
  // double expectedSum = N * ((A * X_VAL) + Y_VAL);
  // double error = sum - expectedSum;
  // printf("Z sum: %llf\n", sum);
  // printf("Expected Z sum: %llf\n", expectedSum);
  // printf("Error: %llf\n", error);

  // // Now free up the results
  // free(z);

  // printf("Done!\n");

  return 0;
}
