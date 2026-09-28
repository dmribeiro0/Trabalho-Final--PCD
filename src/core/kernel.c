/* kernel.c — kernel module implementation.
 *
 * Implements the interface declared in include/kernel.h: built-in kernels,
 * user-defined kernel loading, and any separability handling.
 *
 * TODO: implement this module.
 */
#include "kernel.h"

void get_convolution_filters(Kernel filters[5]) {
  
    for(int f = 0; f < 5; f++) {
        filters[f].size = KERNEL_SIZE;
        for(int i = 0; i < KERNEL_AREA; i++) {
            filters[f].data[i] = 0.0f;
        }
    }

    //coordenadas centrais da matriz
    int cy = KERNEL_SIZE / 2;
    int cx = KERNEL_SIZE / 2;

    if (KERNEL_SIZE >= 3) {
        
        // FILTRO LAPLACIANO (Detecção de Contornos Omnidirecional)
        float laplaciano[3][3] = {
            {-1.0f, -1.0f, -1.0f},
            {-1.0f,  8.0f, -1.0f},
            {-1.0f, -1.0f, -1.0f}
        };

        // FILTRO SOBEL HORIZONTAL (Bordas Horizontais)
        float sobel_h[3][3] = {
            {-1.0f, -2.0f, -1.0f},
            { 0.0f,  0.0f,  0.0f},
            { 1.0f,  2.0f,  1.0f}
        };

        // FILTRO SOBEL VERTICAL (Bordas Verticais)
        float sobel_v[3][3] = {
            {-1.0f,  0.0f,  1.0f},
            {-2.0f,  0.0f,  2.0f},
            {-1.0f,  0.0f,  1.0f}
        };

        // FILTRO SHARPEN (Realce de Nitidez)
        float sharpen[3][3] = {
            { 0.0f, -1.0f,  0.0f},
            {-1.0f,  5.0f, -1.0f},
            { 0.0f, -1.0f,  0.0f}
        };

        // Coloca os filtros no centro da matriz
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pos = (cy + i) * KERNEL_SIZE + (cx + j);
                
                filters[0].data[pos] = laplaciano[i + 1][j + 1];
                filters[1].data[pos] = sobel_h[i + 1][j + 1];
                filters[2].data[pos] = sobel_v[i + 1][j + 1];
                filters[3].data[pos] = sharpen[i + 1][j + 1];
            }
        }
    }
    // FILTRO BOX BLUR (Desfoque de Média)
    // Este filtro expande de acordo com o KERNEL_SIZE
    float peso_blur = 1.0f / KERNEL_AREA;
    for(int i = 0; i < KERNEL_AREA; i++) {
        filters[4].data[i] = peso_blur;
    }
}
