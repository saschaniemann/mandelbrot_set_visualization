#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<chrono>
#include<iostream>

#define N 1000

//Method for adding two Vectors on the CPU
void mat_mul_CPU(float *outCPU, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) {
            float acc = 0.0;

            for(int k = 0; k < n; k++) {
                acc += a[i*n + k] * b[j + k*n];
            }

            outCPU[i*n+j] = acc;
        }
    }
}


// The Method for adding two vectors on the GPU TODO!
__global__ void mat_mul_GPU(float *outGPU, float *a, float *b, int n)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
    {
        outGPU[thread_id] = a[thread_id] + b[thread_id];
    }
}

// // The Method for multiplying two vectors on the GPU TODO!
// __global__ void vector_mul_GPU(float *outGPU, float *a, float *b, int n)
// {
//     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (thread_id < n)
//     {
//         outGPU[thread_id] = a[thread_id] * b[thread_id];
//     }
// }

// // The Method for adding two vectors on the GPU TODO!
// void vector_scalar_CPU(float *outCPU, float *a, float *b, int n)
// {
//     float result;
//     for(int i = 0; i < n; i++)
//     {
//         result += (a[i] * b[i]);
//     }
//     outCPU[0] = result;
// }

// // The Method for adding two vectors on the GPU TODO!
// __global__ void vector_scalar_GPU(float *outGPU, float *a, float *b, int n)
// {
//     __shared__ float temp[N]; // Shared memory to store partial results
//     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (thread_id < n) {
//         temp[threadIdx.x] = a[thread_id] * b[thread_id]; 
//     } else {
//         temp[threadIdx.x] = 0.0f;
//     }
//     __syncthreads(); // Wait for all threads in the block to finish

//     // Perform parallel reduction using shared memory
//     // Reduce the number of active threads by half at each iteration
//     // Each thread adds its partial sum[i] to sum[threadIdx.x + stride]
//     // The last active thread adds the result to outGPU[blockIdx.x]
//     for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (threadIdx.x < stride) {
//             temp[threadIdx.x] += temp[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }

//     // Write the final result to global memory
//     if (threadIdx.x == 0) {
//         atomicAdd(outGPU, temp[0]);
//     }
// }

int main(){
    
    // Declare Vektors on Host and Device
    float *a, *b, *outCPU, *outGPU; 

    // Allocate memory. TODO!
    outCPU 	= (float*)malloc(sizeof(float) * N * N);
    cudaMallocManaged((void**)&a, sizeof(float) * N * N);
    cudaMallocManaged((void**)&b, sizeof(float) * N * N);
    cudaMallocManaged((void**)&outGPU, sizeof(float) * N * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = (float)(((double)rand()) / RAND_MAX); 
	    b[i] = (float)(((double)rand()) / RAND_MAX);
    }
     
    // // Start calculating on CPU:
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    mat_mul_CPU(outCPU, a, b, N);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

    
    /////////////////////////////////////////////////////////////////////////
    // TASKS: 
    // 		1. Transform the vector_add_GPU function to a kernel and call 
    //		   the Kernel.
    //		2. Copy the Memory between Host and Device appropiately
    //		3. Test the implementaion.
    // 		4. Change the  Kernel to be executed by parallel threads
    // 		5. Play around with the number of blocks and thread per block
    /////////////////////////////////////////////////////////////////////////

    // Start calculating on GPU: TODO!
    // Call the Kernel function
    int block_size = 128;
    int grid_size = ((N + block_size -1) / block_size);
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    mat_mul_GPU<<<grid_size,block_size>>>(outGPU, a, b, N);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();


    for(int i = 0; i < N * N; i++)
    {
    	assert(outCPU[i] == outGPU[i]);
    }

    std::cout << "Time on CPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() << "[ms]" << std::endl;
    std::cout << "Time on GPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "[ms]" << std::endl;

	
    // free allocated memory. TODO!
    cudaFree(a);
    cudaFree(b);
    cudaFree(outGPU);
    free(outCPU);
    
}
