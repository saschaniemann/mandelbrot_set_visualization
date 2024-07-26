#include "kernel.cuh"
#include<chrono>
#include<iostream>

__device__ void gradient_to_rgb(float gradient, uint8_t* r, uint8_t* g, uint8_t* b) {
    // related to https://stackoverflow.com/a/6930407 but with v and s set to 1
    double hh, p, q, t, ff;
    int i;

    hh = gradient * 6;
    i = (int)hh;
    ff = hh - i;
    p = 0;
    q = 1.0 - ff;
    t = ff;

    switch(i) {
    case 0:
        *r = (uint8_t) 255;
        *g = (uint8_t) t*255;
        *b = (uint8_t) p*255;
        break;
    case 1:
        *r = (uint8_t) q*255;
        *g = (uint8_t) 255;
        *b = (uint8_t) p*255;
        break;
    case 2:
        *r = (uint8_t) p*255;
        *g = (uint8_t) 255;
        *b = (uint8_t) t*255;
        break;
    case 3:
        *r = (uint8_t) p*255;
        *g = (uint8_t) q*255;
        *b = (uint8_t) 255;
        break;
    case 4:
        *r = (uint8_t) t*255;
        *g = (uint8_t) p*255;
        *b = (uint8_t) 255;
        break;
    case 5:
    default:
        *r = (uint8_t) 255;
        *g = (uint8_t) p*255;
        *b = (uint8_t) q*255;
        break;
    }
}

__global__ void mandelbrot(uint32_t *pixels, int width, int height, float resolution, float offsetX, float offsetY, int numberOfIterations){
    float scale = height / resolution;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    float real = (thread_id % width - (width / 2)) / -scale + offsetX;
    float img = (thread_id / width - (height / 2)) / -scale + offsetY;

    float x = real;
    float tx = x;
    float y = img;
    int n;
    for(n = 0; n < numberOfIterations; n++) {
        // (x, y) = (x^2−y^2−real, 2xy−img)
        tx = x;
        x = x*x - y*y - real;
        y = 2*tx*y - img;
        // if leaving circle with r=2, c=(0,0): break
        if(sqrt(x*x+y*y) > 2) {
            break;
        }
    }
    // in mandelbrot set
    if(n == numberOfIterations) {
        pixels[thread_id] = 0xff000000;
    }
    // not in mandelbrot set
    else {
        uint8_t r, g, b;
        float gradient = ((float) n) / numberOfIterations;
        gradient_to_rgb(gradient, &r, &g, &b);
        // convert rgb to one 32b uint
        pixels[thread_id] = r << 16 | g << 8 | b | 0xff000000;
    }
}

void call_kernel(uint32_t *pixels, int width, int height, float resolution, float offsetX, float offsetY){
    // Allocate memory.
    uint32_t *pixelsGPU; 
    int size = sizeof(uint32_t) * width * height;
    cudaMalloc(&pixelsGPU, size);

    // copy from CPU to GPU
    cudaMemcpy(pixelsGPU, pixels, size, cudaMemcpyHostToDevice);

    // run mandelbrot kernel
    int threadsPerBlock = 256;
    int numberOfBlocks = ((width*height + threadsPerBlock -1) / threadsPerBlock);

    std::chrono::steady_clock::time_point beginKernel = std::chrono::steady_clock::now();
    mandelbrot<<<numberOfBlocks, threadsPerBlock>>>(pixelsGPU, width, height, resolution, offsetX, offsetY, 250);

    // wait for GPU to finish and copy from GPU to CPU
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::cout << "Time for running kernel: " << std::chrono::duration_cast<std::chrono::milliseconds>(endKernel - beginKernel).count() << "[ms]" << std::endl;


    std::chrono::steady_clock::time_point beginCpyToHost = std::chrono::steady_clock::now();
    cudaMemcpy(pixels, pixelsGPU, size, cudaMemcpyDeviceToHost);
    std::chrono::steady_clock::time_point endCpyToHost = std::chrono::steady_clock::now();
    std::cout << "Time for copy to host: " << std::chrono::duration_cast<std::chrono::milliseconds>(endCpyToHost - beginCpyToHost).count() << "[ms]" << std::endl;

    cudaFree(pixelsGPU);
}