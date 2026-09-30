/* convolution.c — convolution module implementation.
 *
 * Implements all convolution variants: sequential, OpenMP, Pthreads.
 * CUDA implementation is in src/parallel/convolution_cuda.cu.
 */

#include "convolution.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

/* Número de threads usado por OpenMP e Pthreads (configurável em runtime). */
static int g_num_threads = 4;

void convolution_set_num_threads(int n) {
    g_num_threads = (n >= 1) ? n : 1;
}

int convolution_get_num_threads(void) {
    return g_num_threads;
}

typedef struct {
    Image *input;
    Kernel *kernel;
    Image *output;
    int start_y;
    int end_y;
} ThreadArgs;

/* Sequential implementation */
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

/* OpenMP implementation - compiles only when _OPENMP is defined */
#ifdef _OPENMP
void apply_convolution_openmp(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;
    int k_size = kernel->size;
    int k_radius = k_size / 2;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * 3 * sizeof(float)); 

    #pragma omp parallel for num_threads(g_num_threads)
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
#else
/* Stub para OpenMP quando _OPENMP não está definido */
void apply_convolution_openmp(Image *input, Kernel *kernel, Image *output) {
    (void)input; (void)kernel; (void)output;
}
#endif

/* Pthreads implementation */
void *pthread_worker(void *args) {
    ThreadArgs *targs = (ThreadArgs *)args;
    int w = targs->input->width;
    int h = targs->input->height;
    int k_size = targs->kernel->size;
    int k_radius = k_size / 2;

    for (int y = targs->start_y; y < targs->end_y; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) { 
                float sum = 0.0f;
                for (int ky = -k_radius; ky <= k_radius; ky++) {
                    for (int kx = -k_radius; kx <= k_radius; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            float pixel_val = targs->input->data[(py * w + px) * 3 + c]; 
                            float weight = targs->kernel->data[(ky + k_radius) * k_size + (kx + k_radius)];
                            sum += pixel_val * weight;
                        }
                    }
                }
                targs->output->data[(y * w + x) * 3 + c] = sum; 
            }
        }
    }
    return NULL;
}

void apply_convolution_pthread(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * 3 * sizeof(float));

    int nt = g_num_threads;
    if (nt < 1) nt = 1;

    pthread_t *threads = (pthread_t *)malloc(nt * sizeof(*threads));
    ThreadArgs *args = (ThreadArgs *)malloc(nt * sizeof(*args));
    if (!threads || !args) {
        fprintf(stderr, "Erro: falha ao alocar estruturas de threads\n");
        free(threads);
        free(args);
        return;
    }

    int chunk_size = h / nt;

    for (int i = 0; i < nt; i++) {
        args[i].input = input;
        args[i].kernel = kernel;
        args[i].output = output;
        args[i].start_y = i * chunk_size;
        args[i].end_y = (i == nt - 1) ? h : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, pthread_worker, &args[i]);
    }

    for (int i = 0; i < nt; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(args);
}

/* Stub para CUDA (implementado em src/parallel/convolution_cuda.cu) */
#ifndef HAVE_CUDA
void apply_convolution_cuda(Image *input, Kernel *kernel, Image *output) {
    (void)input; (void)kernel; (void)output;
    fprintf(stderr, "Erro: CUDA não implementado ou não disponível\n");
    exit(1);
}
#endif
