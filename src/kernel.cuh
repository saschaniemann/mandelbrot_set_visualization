#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<chrono>
#include<iostream>

__device__ void gradient_to_rgb(float gradient, unsigned char* r, unsigned char* g, unsigned char* b);
__global__ void mandelbrot(uint32_t *pixels, int width, int height);
void call_kernel(uint32_t *pixels, int width, int height, float resolution, float offsetX, float offsetY);