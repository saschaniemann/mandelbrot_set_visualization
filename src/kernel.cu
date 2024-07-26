#include "kernel.cuh"
#include<chrono>
#include<iostream>

__device__ void gradient_to_rgb(float gradient, unsigned char* r, unsigned char* g, unsigned char* b) {
    // hsv to rgb with h being the gradient and v and s fixed
    float h = gradient;
    float v = 0.8;
    float s = 1.0;
    int i = (int)(h * 6);
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: *r = (unsigned char)(v * 255); *g = (unsigned char)(t * 255); *b = (unsigned char)(p * 255); break;
        case 1: *r = (unsigned char)(q * 255); *g = (unsigned char)(v * 255); *b = (unsigned char)(p * 255); break;
        case 2: *r = (unsigned char)(p * 255); *g = (unsigned char)(v * 255); *b = (unsigned char)(t * 255); break;
        case 3: *r = (unsigned char)(p * 255); *g = (unsigned char)(q * 255); *b = (unsigned char)(v * 255); break;
        case 4: *r = (unsigned char)(t * 255); *g = (unsigned char)(p * 255); *b = (unsigned char)(v * 255); break;
        case 5: *r = (unsigned char)(v * 255); *g = (unsigned char)(p * 255); *b = (unsigned char)(q * 255); break;
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
        tx = x;
        x = x*x - y*y - real;
        y = 2*tx*y - img;
        // (x2−y2−a, 2xy−b)
        if(sqrt(x*x+y*y) > 2) {
            break;
        }
    }
    // not in mandelbrot set
    if(n == numberOfIterations) {
        pixels[thread_id] = 0xff000000;
    }
    else {
        unsigned char r, g, b;
        float gradient = ((float) n) / numberOfIterations;
        gradient_to_rgb(gradient, &r, &g, &b);

        pixels[thread_id] = r << 16 | g << 8 | b | 0xff000000;
    }
}

void call_kernel(uint32_t *pixels, int width, int height, float resolution, float offsetX, float offsetY){
    // Declare Vektors on Host and Device
    uint32_t *pixelsGPU; 

    // Allocate memory.
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