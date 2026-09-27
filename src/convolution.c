/* convolution.c — convolution module implementation.
 *
 * Implements the interface declared in include/convolution.h: the general and
 * separable convolution routines. This is the compute core to be parallelized
 * in later phases (OpenMP / Pthreads / CUDA).
 *
 * TODO: implement this module.
 */
#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>

void apply_convolution_sequential(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;
    int k_size = kernel->size;
    int k_radius = k_size / 2;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * sizeof(float));

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

void get_convolution_filters_sequential(Kernel filters[5]) {
  
    // Filtro Detecção de Borda Horizontal
    filters[1].size = 3;
    float f1[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    for(int i=0; i<9; i++) filters[1].data[i] = f1[i];

    // Filtro Detecção de Borda Vertical
    filters[2].size = 3;
    float f2[9] = {
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    };
    for(int i=0; i<9; i++) filters[2].data[i] = f2[i];

    // Filtro Sharpening (Realce de Detalhes)
    filters[3].size = 3;
    float f3[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
    for(int i=0; i<9; i++) filters[3].data[i] = f3[i];

    // Filtro Box Blur (Desfoque de Média)
    filters[4].size = 3;
    float f4[9] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    for(int i=0; i<9; i++) filters[4].data[i] = f4[i];
}
