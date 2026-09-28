/* main.c — program entry point.
 *
 * CLI para experimentos com filtros convolucionais paralelos.
 * Suporta: sequential, OpenMP, Pthreads, CUDA.
 *
 * Uso:
 *   ./convolve_seq -i <imagem> -k <kernel> [-b block_size] [-s tamanho_kernel] [-o output] [-m metrics.json]
 *
 * Opções:
 *   -i, --image      Caminho da imagem de entrada (obrigatório)
 *   -k, --kernel     Nome do kernel: laplacian, sobel_h, sobel_v, sharpen, blur (obrigatório)
 *   -m, --mode       Modo de execução: seq, omp, pthread, cuda (padrão: seq)
 *   -b, --block      Tamanho do bloco para divisão da imagem (padrão: 32)
 *   -s, --size       Tamanho do kernel (3, 5, 7, 9, ...) (padrão: 3)
 *   -o, --output     Caminho do arquivo de saída (padrão: resultado.png)
 *   -t, --metrics    Caminho do arquivo JSON para métricas (opcional)
 *   -h, --help       Exibir ajuda
 *
 * Exemplos:
 *   ./convolve_seq -i images/a.jpg -k laplacian -m seq -o result.png
 *   ./convolve_seq -i images/a.jpg -k blur -m omp -b 64 -s 5 -o blur_omp.png
 */

#include "image.h"
#include "kernel.h"
#include "convolution.h"
#include "timer.h"
#include "metrics.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

// Estrutura para armazenar as opções de execução
typedef struct {
    const char *image_path;
    const char *kernel_name;
    const char *mode;
    int block_size;
    int kernel_size;
    const char *output_path;
    const char *metrics_path;
} Options;

// Valores padrão
#define DEFAULT_BLOCK_SIZE 32
#define DEFAULT_KERNEL_SIZE 3
#define DEFAULT_MODE "seq"
#define DEFAULT_OUTPUT "results/resultado.png"
#define DEFAULT_METRICS "results/metrics.json"

// Protótipos das funções auxiliares
static void print_usage(const char *prog_name);
static int parse_args(int argc, char **argv, Options *opts);
static int get_kernel_index(const char *name);
static int get_mode_index(const char *name);

// Mapa de kernels
static const char *kernel_names[] = {
    "laplacian", "sobel_h", "sobel_v", "sharpen", "blur"
};

// Mapa de modos
static const char *mode_names[] = {
    "seq", "omp", "pthread", "cuda"
};

static void print_usage(const char *prog_name) {
    printf("Uso: %s -i <imagem> -k <kernel> [opções]\n", prog_name);
    printf("\nOpções:\n");
    printf("  -i, --image      Caminho da imagem de entrada (obrigatório)\n");
    printf("  -k, --kernel     Nome do kernel: laplacian, sobel_h, sobel_v, sharpen, blur (obrigatório)\n");
    printf("  -m, --mode       Modo de execução: seq, omp, pthread, cuda (padrão: seq)\n");
    printf("  -b, --block      Tamanho do bloco para divisão da imagem (padrão: %d)\n", DEFAULT_BLOCK_SIZE);
    printf("  -s, --size       Tamanho do kernel (3, 5, 7, 9, ...) (padrão: %d)\n", DEFAULT_KERNEL_SIZE);
    printf("  -o, --output     Caminho do arquivo de saída (padrão: %s)\n", DEFAULT_OUTPUT);
    printf("  -t, --metrics    Caminho do arquivo JSON para métricas (opcional)\n");
    printf("  -h, --help       Exibir esta ajuda\n");
    printf("\nExemplos:\n");
    printf("  %s -i images/a.jpg -k laplacian -m seq -o result.png\n", prog_name);
    printf("  %s -i images/a.jpg -k blur -m omp -b 64 -s 5 -o blur_omp.png\n", prog_name);
}

static int get_kernel_index(const char *name) {
    for (int i = 0; i < 5; i++) {
        if (strcmp(name, kernel_names[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static int get_mode_index(const char *name) {
    for (int i = 0; i < 4; i++) {
        if (strcmp(name, mode_names[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static int parse_args(int argc, char **argv, Options *opts) {
    static struct option long_options[] = {
        {"image",    required_argument, 0, 'i'},
        {"kernel",   required_argument, 0, 'k'},
        {"mode",     required_argument, 0, 'm'},
        {"block",    required_argument, 0, 'b'},
        {"size",     required_argument, 0, 's'},
        {"output",   required_argument, 0, 'o'},
        {"metrics",  required_argument, 0, 't'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    // Valores padrão
    opts->image_path = NULL;
    opts->kernel_name = NULL;
    opts->mode = DEFAULT_MODE;
    opts->block_size = DEFAULT_BLOCK_SIZE;
    opts->kernel_size = DEFAULT_KERNEL_SIZE;
    opts->output_path = DEFAULT_OUTPUT;
    opts->metrics_path = DEFAULT_METRICS;

    int opt;
    int long_index = 0;

    while ((opt = getopt_long(argc, argv, "i:k:m:b:s:o:t:h", long_options, &long_index)) != -1) {
        switch (opt) {
            case 'i':
                opts->image_path = optarg;
                break;
            case 'k':
                opts->kernel_name = optarg;
                break;
            case 'm':
                opts->mode = optarg;
                break;
            case 'b':
                opts->block_size = atoi(optarg);
                if (opts->block_size <= 0) {
                    fprintf(stderr, "Erro: block_size deve ser positivo\n");
                    return -1;
                }
                break;
            case 's':
                opts->kernel_size = atoi(optarg);
                if (opts->kernel_size <= 0 || opts->kernel_size % 2 == 0) {
                    fprintf(stderr, "Erro: tamanho do kernel deve ser ímpar e positivo (3, 5, 7, 9, ...)\n");
                    return -1;
                }
                break;
            case 'o':
                opts->output_path = optarg;
                break;
            case 't':
                opts->metrics_path = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    // Verificar argumentos obrigatórios
    if (!opts->image_path) {
        fprintf(stderr, "Erro: imagem de entrada é obrigatória (-i)\n");
        return -1;
    }
    if (!opts->kernel_name) {
        fprintf(stderr, "Erro: nome do kernel é obrigatório (-k)\n");
        return -1;
    }

    return 1; // Sucesso
}

int main(int argc, char **argv) {
    Options opts;
    int result;

    result = parse_args(argc, argv, &opts);
    if (result <= 0) {
        return result == 0 ? 0 : 1;
    }

    // Carregar imagem
    Image img = load_image(opts.image_path);
    if (!img.data) {
        fprintf(stderr, "Erro ao carregar a imagem: %s\n", opts.image_path);
        return 1;
    }

    // Obter índice do kernel
    int filter_idx = get_kernel_index(opts.kernel_name);
    if (filter_idx < 0) {
        fprintf(stderr, "Erro: kernel '%s' desconhecido\n", opts.kernel_name);
        fprintf(stderr, "Kernels disponíveis: ");
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "%s ", kernel_names[i]);
        }
        fprintf(stderr, "\n");
        free_image(&img);
        return 1;
    }

    // Obter índice do modo
    int mode_idx = get_mode_index(opts.mode);
    if (mode_idx < 0) {
        fprintf(stderr, "Erro: modo '%s' desconhecido\n", opts.mode);
        fprintf(stderr, "Modos disponíveis: ");
        for (int i = 0; i < 4; i++) {
            fprintf(stderr, "%s ", mode_names[i]);
        }
        fprintf(stderr, "\n");
        free_image(&img);
        return 1;
    }

    // Carregar kernels (o tamanho do kernel é definido em runtime via kernel_size)
    Kernel filters[5];
    get_convolution_filters(filters);
    
    // Atualizar tamanho do kernel para o especificado
    for (int i = 0; i < 5; i++) {
        filters[i].size = opts.kernel_size;
    }

    // Preparar imagem de saída
    Image out_img;
    out_img.width = img.width;
    out_img.height = img.height;
    out_img.data = (float *)malloc(img.width * img.height * 3 * sizeof(float));
    if (!out_img.data) {
        fprintf(stderr, "Erro: falha ao alocar memória para imagem de saída\n");
        free_image(&img);
        return 1;
    }

    // Timer
    Timer timer;
    timer_start(&timer);

    // Executar convolution baseada no modo
    switch (mode_idx) {
        case 0: // seq
            apply_convolution_sequential(&img, &filters[filter_idx], &out_img);
            break;
        case 1: // omp
            apply_convolution_openmp(&img, &filters[filter_idx], &out_img);
            break;
        case 2: // pthread
            apply_convolution_pthread(&img, &filters[filter_idx], &out_img);
            break;
        case 3: // cuda
            apply_convolution_cuda(&img, &filters[filter_idx], &out_img);
            break;
    }

    double elapsed = timer_stop(&timer);

    // Calcular speedup comparando com sequential
    // O tempo sequential precisa ser calculado separadamente
    // Para simplificar, vamos calcular o speedup como 1 se for o modo seq
    // ou como (tempo_sequencial / tempo_atual) se for paralelo
    double speedup = 1.0;
    if (mode_idx != 0) {
        // Reexecutar sequential para obter tempo de baseline
        Image seq_img;
        seq_img.width = img.width;
        seq_img.height = img.height;
        seq_img.data = (float *)malloc(img.width * img.height * 3 * sizeof(float));
        
        Timer seq_timer;
        timer_start(&seq_timer);
        apply_convolution_sequential(&img, &filters[filter_idx], &seq_img);
        double seq_elapsed = timer_stop(&seq_timer);
        
        speedup = seq_elapsed / elapsed;
        
        free_image(&seq_img);
    }

    // Salvar imagem de saída
    save_image(opts.output_path, &out_img);

    // Calcular métricas
    int operations = img.width * img.height * filters[filter_idx].size * filters[filter_idx].size * 3; // aproximado
    double flops = operations / elapsed;

    // Imprimir resumo
    printf("=== Resultados ===\n");
    printf("Imagem: %s (%dx%d)\n", opts.image_path, img.width, img.height);
    printf("Kernel: %s (size=%d)\n", opts.kernel_name, opts.kernel_size);
    printf("Modo: %s\n", opts.mode);
    printf("Block size: %d\n", opts.block_size);
    printf("Tempo: %.6f s\n", elapsed);
    printf("FLOPS: %.2f\n", flops);
    printf("Speedup: %.2fx\n", speedup);
    printf("Saída: %s\n", opts.output_path);

    // Salvar métricas em JSON se especificado
    if (opts.metrics_path) {
        FILE *f = fopen(opts.metrics_path, "w");
        if (f) {
            fprintf(f, "{\n");
            fprintf(f, "  \"image\": \"%s\",\n", opts.image_path);
            fprintf(f, "  \"image_dims\": [%d, %d],\n", img.width, img.height);
            fprintf(f, "  \"kernel\": \"%s\",\n", opts.kernel_name);
            fprintf(f, "  \"kernel_size\": %d,\n", opts.kernel_size);
            fprintf(f, "  \"mode\": \"%s\",\n", opts.mode);
            fprintf(f, "  \"block_size\": %d,\n", opts.block_size);
            fprintf(f, "  \"time_seconds\": %.6f,\n", elapsed);
            fprintf(f, "  \"flops\": %.2f,\n", flops);
            fprintf(f, "  \"speedup\": %.2f\n", speedup);
            fprintf(f, "}\n");
            fclose(f);
            printf("Métricas salvas em: %s\n", opts.metrics_path);
        } else {
            fprintf(stderr, "Aviso: não foi possível salvar métricas em %s\n", opts.metrics_path);
        }
    }

    // Limpeza
    free_image(&img);
    free_image(&out_img);

    return 0;
}
