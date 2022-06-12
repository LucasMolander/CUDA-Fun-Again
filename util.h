#pragma once

#define CUDA_SUCC cudaError::cudaSuccess
#define H_TO_D cudaMemcpyKind::cudaMemcpyHostToDevice
#define D_TO_H cudaMemcpyKind::cudaMemcpyDeviceToHost

int printDeviceInfo();

int getDeviceProps(cudaDeviceProp *props);
