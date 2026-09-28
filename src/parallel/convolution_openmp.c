/* convolution_openmp.c — OpenMP implementation of 2D convolution.
 *
 * Implements apply_convolution_openmp() using #pragma omp parallel for.
 */

#include "convolution.h"
#include <stdlib.h>

void apply_convolution_openmp(Image *input, Kernel *kernel, Image *output) {
    int w = input->width;
    int h = input->height;
    int k_size = kernel->size;
    int k_radius = k_size / 2;

    output->width = w;
    output->height = h;
    output->data = (float *)malloc(w * h * 3 * sizeof(float)); 

    #pragma omp parallel for
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
