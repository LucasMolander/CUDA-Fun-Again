#include <stdlib.h>
#include <stdio.h>

#define N 20'000'000

int main(void) {
  printf("main()\n");

  float *x, *y;     // Host arrays
  float *d_x, *d_y; // Device arrays

  // Allocate space
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));

  // Set the values
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  printf("Hello, world!\n");

  return 0;
}
