#include "convolution.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ static void convolve(const float *input, const float *weights,
                                float *output, int width, int height, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = size / 2;
    for (int channel = 0; channel < 3; channel++) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int px = x + kx;
                int py = y + ky;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    sum += input[(py * width + px) * 3 + channel] *
                           weights[(ky + radius) * size + kx + radius];
                }
            }
        }
        output[(y * width + x) * 3 + channel] = sum;
    }
}

static void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

extern "C" void apply_convolution_cuda(Image *input, Kernel *kernel, Image *output) {
    if (kernel->size != KERNEL_SIZE) {
        fprintf(stderr, "CUDA: tamanho de kernel suportado: %d\n", KERNEL_SIZE);
        exit(EXIT_FAILURE);
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "verificando GPU");
    if (device_count == 0) {
        fprintf(stderr, "CUDA: nenhuma GPU disponivel\n");
        exit(EXIT_FAILURE);
    }

    size_t image_bytes = (size_t)input->width * input->height * 3 * sizeof(float);
    size_t kernel_bytes = (size_t)kernel->size * kernel->size * sizeof(float);

    /* Aloca a saída internamente (mesmo contrato dos backends seq/omp/pthread:
     * cada apply_convolution_* aloca output->data). */
    output->width = input->width;
    output->height = input->height;
    output->data = (float *)malloc(image_bytes);
    if (!output->data) {
        fprintf(stderr, "CUDA: falha ao alocar imagem de saida no host\n");
        exit(EXIT_FAILURE);
    }

    float *device_input, *device_weights, *device_output;
    check_cuda(cudaMalloc((void **)&device_input, image_bytes), "alocando entrada");
    check_cuda(cudaMalloc((void **)&device_weights, kernel_bytes), "alocando filtro");
    check_cuda(cudaMalloc((void **)&device_output, image_bytes), "alocando saida");
    check_cuda(cudaMemcpy(device_input, input->data, image_bytes, cudaMemcpyHostToDevice), "copiando entrada");
    check_cuda(cudaMemcpy(device_weights, kernel->data, kernel_bytes, cudaMemcpyHostToDevice), "copiando filtro");

    dim3 threads(16, 16);
    dim3 blocks((input->width + 15) / 16, (input->height + 15) / 16);
    convolve<<<blocks, threads>>>(device_input, device_weights, device_output,
                                   input->width, input->height, kernel->size);
    check_cuda(cudaGetLastError(), "iniciando convolucao");
    check_cuda(cudaDeviceSynchronize(), "executando convolucao");
    check_cuda(cudaMemcpy(output->data, device_output, image_bytes, cudaMemcpyDeviceToHost), "copiando saida");

    check_cuda(cudaFree(device_input), "liberando entrada");
    check_cuda(cudaFree(device_weights), "liberando filtro");
    check_cuda(cudaFree(device_output), "liberando saida");
}
