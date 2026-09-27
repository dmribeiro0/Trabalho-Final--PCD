#include "../include/convolution.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void convolution_kernel_cuda(float *in_data, float *out_data, float *kernel_data, int w, int h, int k_size, int k_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        for (int c = 0; c < 3; c++) { 
            float sum = 0.0f;
            for (int ky = -k_radius; ky <= k_radius; ky++) {
                for (int kx = -k_radius; kx <= k_radius; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        float pixel_val = in_data[(py * w + px) * 3 + c]; // Índice ajustado
                        float weight = kernel_data[(ky + k_radius) * k_size + (kx + k_radius)];
                        sum += pixel_val * weight;
                    }
                }
            }
            out_data[(y * w + x) * 3 + c] = sum; 
        }
    }
}

void apply_convolution_cuda(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;
    int k_size = kernel->size;
    int k_radius = k_size / 2;

    output->width = w;
    output->height = h;
    size_t img_size = w * h * 3 * sizeof(float); 
    size_t k_bytes = k_size * k_size * sizeof(float);
    output->data = (float *)malloc(img_size);

    float *d_in, *d_out, *d_kernel;
    cudaMalloc((void **)&d_in, img_size);
    cudaMalloc((void **)&d_out, img_size);
    cudaMalloc((void **)&d_kernel, k_bytes);

    cudaMemcpy(d_in, input->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel->data, k_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolution_kernel_cuda<<<numBlocks, threadsPerBlock>>>(d_in, d_out, d_kernel, w, h, k_size, k_radius);

    cudaMemcpy(output->data, d_out, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
}