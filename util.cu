#include <stdio.h>

#include "util.h"

int printDeviceInfo() {
  cudaError e;

  int nDevices;
  if ((e = cudaGetDeviceCount(&nDevices)) != CUDA_SUCC) {
    printf("Failed to get the device count: %d\n", e);
    return 1;
  }

  printf("N devices: %d\n", nDevices);

  cudaDeviceProp prop;
  for (int i = 0; i < nDevices; i++) {
    if ((e = cudaGetDeviceProperties(&prop, i)) != CUDA_SUCC) {
      printf("Failed to get device properties[%d]: %d\n", i, e);
      return 2;
    }

    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf(
      "  Peak Memory Bandwidth (GB/s): %f\n",
      2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6
    );
    printf("  Warp Size: %d\n\n", prop.warpSize);
  }

  return 0;
}

int getDeviceProps(cudaDeviceProp *props) {
  cudaError e;

  int nDevices;
  if ((e = cudaGetDeviceCount(&nDevices)) != CUDA_SUCC) {
    printf("Failed to get the device count: %d\n", e);
    return -1;
  }

  if (nDevices < 1) {
    printf("%d devices! No can do.\n", e);
    return -1;
  }

  int deviceIdx = 0;
  if ((e = cudaGetDeviceProperties(props, deviceIdx)) != CUDA_SUCC) {
    printf("Failed to get device properties[%d]: %d\n", deviceIdx, e);
    return -1;
  }

  return 0;
}
