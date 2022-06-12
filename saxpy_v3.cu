#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "util.h"

/**
This is 13.02 billion daxpys per second for N = 200,000,000

It's for doubles rather than floats
*/

#define N 200'000'000
#define TOTAL_SIZE N * sizeof(double)

#define A 2.3f
#define X_VAL 1.0f
#define Y_VAL 2.0f


__global__
void daxpy(uint64_t n, double a, double *d_x, double *d_y, double *d_z) {
  uint64_t tIdx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tIdx < n) {
    d_z[tIdx] = (a * d_x[tIdx]) + d_y[tIdx];
  }
}


int main(void) {
  printf("SAXPY Version 3 (it's really a DAXPY)\n");

  printf("Total size of a vector: %lld MB\n", TOTAL_SIZE / (1024 * 1024));

  cudaDeviceProp props;
  if (getDeviceProps(&props) != 0) {
    printf("Unable to get device props\n");
    return 1;
  }

  cudaError e;
  double *x, *y, *z;       // Host arrays
  double *d_x, *d_y, *d_z; // Device arrays

  // Allocate space on Host
  x = (double*)malloc(TOTAL_SIZE);
  y = (double*)malloc(TOTAL_SIZE);
  z = (double*)malloc(TOTAL_SIZE);

  // Allocate space on Device
  if ((e = cudaMalloc(&d_x, TOTAL_SIZE)) != CUDA_SUCC) {
    printf("Failed to allocate device X: %d\n", e);
    return 1;
  }
  if ((e = cudaMalloc(&d_y, TOTAL_SIZE)) != CUDA_SUCC) {
    printf("Failed to allocate device Y: %d\n", e);
    return 1;
  }
  if ((e = cudaMalloc(&d_z, TOTAL_SIZE)) != CUDA_SUCC) {
    printf("Failed to allocate device Z: %d\n", e);
    return 1;
  }

  // Set the values on Host (Z is for the result)
  for (uint64_t i = 0; i < N; i++) {
    x[i] = X_VAL;
    y[i] = Y_VAL;
  }

  // Copy values for X and Y over to Device (Z is for the result)
  if ((e = cudaMemcpy(d_x, x, TOTAL_SIZE, H_TO_D)) != CUDA_SUCC) {
    printf("Failed to copy X to device: %d\n", e);
    return 1;
  }
  if ((e = cudaMemcpy(d_y, y, TOTAL_SIZE, H_TO_D)) != CUDA_SUCC) {
    printf("Failed to copy Y to device: %d\n", e);
    return 1;
  }

  // Block size
  int warpSize = props.warpSize;
  uint64_t maxBlockSizeX = props.maxThreadsDim[0];
  int blockSize = warpSize * (maxBlockSizeX / warpSize);

  // Grid size
  uint64_t maxGridSizeX = props.maxGridSize[0];
  uint64_t nBlocks = N / blockSize;
  int nExtraThreads = N % blockSize;
  if (nExtraThreads > 0) {
    ++nBlocks;
  }

  printf("N: %lld\n", (uint64_t)N);
  printf("Warp size: %d\n", warpSize);
  printf("Max block size X: %lld\n", maxBlockSizeX);
  printf("Block size: %d\n", blockSize);
  printf("Max grid size X: %lld\n", maxGridSizeX);
  printf("N blocks: %lld\n", nBlocks);
  printf("N extra threads (last block): %d\n", nExtraThreads);

  if (nBlocks > maxGridSizeX) {
    printf("nBlocks > maxGridSizeX (%lld > %lld)\n", nBlocks, maxGridSizeX);
    return 1;
  }

  printf("A: %f\n", A);

  daxpy<<<nBlocks, blockSize>>>(N, A, d_x, d_y, d_z);

  // Copy Z back to Host
  if ((e = cudaMemcpy(z, d_z, TOTAL_SIZE, D_TO_H)) != CUDA_SUCC) {
    printf("Failed to copy Z to host: %d\n", e);
    return 1;
  }

  // Free up the Device memory
  if ((e = cudaFree(d_x)) != CUDA_SUCC) {
    printf("Failed to free device X: %d", e);
    return 1;
  }
  if ((e = cudaFree(d_y)) != CUDA_SUCC) {
    printf("Failed to free device Y: %d", e);
    return 1;
  }
  if ((e = cudaFree(d_z)) != CUDA_SUCC) {
    printf("Failed to free device Z: %d", e);
    return 1;
  }

  // Free up the host memory
  free(x);
  free(y);

  // Do some stuff with the results
  double sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += z[i];
  }
  double expectedSum = N * ((A * X_VAL) + Y_VAL);
  double error = sum - expectedSum;
  printf("Z sum: %llf\n", sum);
  printf("Expected Z sum: %llf\n", expectedSum);
  printf("Error: %llf\n", error);

  // Now free up the results
  free(z);

  printf("Done!\n");

  return 0;
}
