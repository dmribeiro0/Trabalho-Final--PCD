/* main.c — program entry point.
 *
 * Wires the pipeline together: parse CLI arguments, load an image, select a
 * kernel, run the convolution (timed), and save the result.
 *
 * TODO: implement argument parsing and pipeline wiring.
 */
#include "matrix.h"
#include "timer.h"
#include "metrics.h"
#include "../include/image.h"
#include "../include/kernel.h"
#include "../include/convolution.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// int main(void) {
//     return 0;
// }


// FUNCAO TESTEEEE, APENAS PRA VER SE TA FUNCIONANDO OS CODIGOS, SEM OLHAR METRICAS AINDA!!!!!!! - CLARA E RAFA

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("IMAGEM: %s\n", argv[0]);
        return 1;
    }

    Image img = load_image(argv[1]);
    if (!img.data) {
        printf("Erro ao carregar a imagem.\n");
        return 0;
    }

    int modo_idx;
    printf("Modos de execucao:\n");
    printf("0 - Sequencial (seq)\n");
    printf("1 - OpenMP (omp)\n");
    printf("2 - Pthreads (pth)\n");
    printf("3 - CUDA (cuda)\n");
    printf("Escolha o modo: ");
    scanf("%d", &modo_idx);

    if (modo_idx < 0 || modo_idx > 3) {
        printf("Modo invalido. Usando 0 (seq) por padrao.\n");
        modo_idx = 0;
    }

    int filter_idx;
    printf("\nFiltros:\n");
    printf("0 - Laplaciano\n");
    printf("1 - Sobel Horizontal\n");
    printf("2 - Sobel Vertical\n");
    printf("3 - Sharpen (Realce)\n");
    printf("4 - Box Blur (Desfoque)\n");
    printf("Escolha o filtro: ");
    scanf("%d", &filter_idx);

    if (filter_idx < 0 || filter_idx > 4) {
        printf("Filtro invalido. Usando 0 por padrao.\n");
        filter_idx = 0;
    }

    Kernel filters[5];
    get_convolution_filters(filters);

    Image out_img;
    const char *modo_nomes[] = {"seq", "omp", "pth", "cuda"};

    if (modo_idx == 0) {
        apply_convolution_sequential(&img, &filters[filter_idx], &out_img);
    } else if (modo_idx == 1) {
        apply_convolution_openmp(&img, &filters[filter_idx], &out_img);
    } else if (modo_idx == 2) {
        apply_convolution_pthread(&img, &filters[filter_idx], &out_img);
    } else if (modo_idx == 3) {
        apply_convolution_cuda(&img, &filters[filter_idx], &out_img);
    }

    char out_filename[256];
    sprintf(out_filename, "resultado_%s_f%d.png", modo_nomes[modo_idx], filter_idx);
    save_image(out_filename, &out_img);

    free_image(&img);
    free_image(&out_img);

    return 0;
}
