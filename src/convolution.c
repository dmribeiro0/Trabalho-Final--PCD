/* convolution.c — convolution module implementation.
 *
 * Implements the interface declared in include/convolution.h: the general and
 * separable convolution routines. This is the compute core to be parallelized
 * in later phases (OpenMP / Pthreads / CUDA).
 *
 * TODO: implement this module.
 */
#include "../include/convolution.h"
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4
typedef struct {
    Image *input;
    Kernel *kernel;
    Image *output;
    int start_y;
    int end_y;
} ThreadArgs;

void apply_convolution_sequential(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;
    int k_size = kernel->size;
    int k_radius = k_size / 2;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * 3 * sizeof(float)); 

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0.0f;

                for (int ky = -k_radius; ky <= k_radius; ky++) {
                    for (int kx = -k_radius; kx <= k_radius; kx++) {
                        int px = x + kx;
                        int py = y + ky;

                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            float pixel_val = input->data[(py * w + px) * 3 + c];
                            float weight = kernel->data[(ky + k_radius) * k_size + (kx + k_radius)];
                            sum += pixel_val * weight;
                        }
                    }
                }
                output->data[(y * w + x) * 3 + c] = sum;
            }
        }
    }
}

void apply_convolution_openmp(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;
    int k_size = kernel->size;
    int k_radius = k_size / 2;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * sizeof(float));

    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;

            for (int ky = -k_radius; ky <= k_radius; ky++) {
                for (int kx = -k_radius; kx <= k_radius; kx++) {
                    int px = x + kx;
                    int py = y + ky;

                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        float pixel_val = input->data[py * w + px];
                        float weight = kernel->data[(ky + k_radius) * k_size + (kx + k_radius)];
                        sum += pixel_val * weight;
                    }
                }
            }
            output->data[y * w + x] = sum;
        }
    }
}

void *pthread_worker(void *args) {
    ThreadArgs *targs = (ThreadArgs *)args;
    int w = targs->input->width;
    int h = targs->input->height;
    int k_size = targs->kernel->size;
    int k_radius = k_size / 2;

    for (int y = targs->start_y; y < targs->end_y; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int ky = -k_radius; ky <= k_radius; ky++) {
                for (int kx = -k_radius; kx <= k_radius; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        float pixel_val = targs->input->data[py * w + px];
                        float weight = targs->kernel->data[(ky + k_radius) * k_size + (kx + k_radius)];
                        sum += pixel_val * weight;
                    }
                }
            }
            targs->output->data[y * w + x] = sum;
        }
    }
    return NULL;
}

void apply_convolution_pthread(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * sizeof(float));

    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    int chunk_size = h / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].input = input;
        args[i].kernel = kernel;
        args[i].output = output;
        args[i].start_y = i * chunk_size;
        args[i].end_y = (i == NUM_THREADS - 1) ? h : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, pthread_worker, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}
