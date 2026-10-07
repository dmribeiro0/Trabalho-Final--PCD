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
#include "energy.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>

// Estrutura para armazenar as opções de execução
typedef struct {
    const char *image_path;
    const char *kernel_name;
    const char *mode;
    int block_size;
    int kernel_size;
    int threads;
    const char *output_path;
    const char *metrics_path;
} Options;

// Valores padrão
#define DEFAULT_BLOCK_SIZE 32
#define DEFAULT_KERNEL_SIZE 3
#define DEFAULT_THREADS 4
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
    printf("  -p, --threads    Número de threads (omp/pthread; padrão: %d)\n", DEFAULT_THREADS);
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
        {"threads",  required_argument, 0, 'p'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    // Valores padrão
    opts->image_path = NULL;
    opts->kernel_name = NULL;
    opts->mode = DEFAULT_MODE;
    opts->block_size = DEFAULT_BLOCK_SIZE;
    opts->kernel_size = DEFAULT_KERNEL_SIZE;
    opts->threads = DEFAULT_THREADS;
    opts->output_path = DEFAULT_OUTPUT;
    opts->metrics_path = DEFAULT_METRICS;

    int opt;
    int long_index = 0;

    while ((opt = getopt_long(argc, argv, "i:k:m:b:s:o:t:p:h", long_options, &long_index)) != -1) {
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
            case 'p':
                opts->threads = atoi(optarg);
                if (opts->threads < 1) {
                    fprintf(stderr, "Erro: número de threads deve ser >= 1\n");
                    return -1;
                }
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

    // Preparar imagem de saída (a convolução aloca out_img.data internamente).
    Image out_img;
    out_img.width = img.width;
    out_img.height = img.height;
    out_img.data = NULL;

    // Configura o número de threads (afeta omp/pthread).
    convolution_set_num_threads(opts.threads);

    // P = número de threads: seq usa 1, omp/pthread usam -p; CUDA não usa P.
    int p_value = (mode_idx == 0) ? 1 : opts.threads;

    // Timer
    Timer timer;

    // Amostra de energia (RAPL) antes da região cronometrada.
    EnergySample e_begin = energy_read();
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
    EnergySample e_end = energy_read();

    // Energia consumida na região cronometrada (J) ou METRIC_NA.
    double energy_j = energy_delta_joules(&e_begin, &e_end);
    if (energy_j < 0.0) {
        energy_j = METRIC_NA;
    }

    // Speedup(P) = T_s / T_p. Para o modo seq, T_s == T_p => speedup 1.
    // Para os modos paralelos, reexecuta o sequencial como baseline T_s.
    double speedup = METRIC_NA;
    if (mode_idx == 0) {
        speedup = 1.0;
    } else {
        Image seq_img;
        seq_img.width = img.width;
        seq_img.height = img.height;
        seq_img.data = NULL;

        Timer seq_timer;
        timer_start(&seq_timer);
        apply_convolution_sequential(&img, &filters[filter_idx], &seq_img);
        double seq_elapsed = timer_stop(&seq_timer);

        speedup = metrics_speedup(seq_elapsed, elapsed);
        free_image(&seq_img);
    }

    // Eficiência = speedup / P. Não definida para CUDA (P não corresponde a threads de CPU).
    double efficiency;
    if (mode_idx == 3) {
        efficiency = METRIC_NA;
    } else {
        efficiency = metrics_efficiency(speedup, p_value);
    }

    // Salvar imagem de saída
    // Forçar caminho relativo à pasta results/
    char final_output_path[1024];
    if (opts.output_path[0] != '/') {
        snprintf(final_output_path, sizeof(final_output_path), "results/%s", opts.output_path);
    } else {
        strncpy(final_output_path, opts.output_path, sizeof(final_output_path));
    }
    // Criar diretório de saída se não existir
    char *last_slash = strrchr(final_output_path, '/');
    if (last_slash) {
        char dir[512];
        size_t dir_len = last_slash - final_output_path;
        strncpy(dir, final_output_path, dir_len);
        dir[dir_len] = '\0';
        mkdir(dir, 0755);
    }
    save_image(final_output_path, &out_img);

    // Calcular métricas usando o módulo de métricas.
    double flops = metrics_flops(img.width, img.height, 3, opts.kernel_size, elapsed);
    double mflops = metrics_mflops(flops);
    double avg_power = metrics_avg_power(energy_j, elapsed);
    double mflops_per_watt = metrics_mflops_per_watt(mflops, avg_power);

    // Imprimir resumo
    printf("=== Resultados ===\n");
    printf("Imagem: %s (%dx%d)\n", opts.image_path, img.width, img.height);
    printf("Kernel: %s (size=%d)\n", opts.kernel_name, opts.kernel_size);
    printf("Modo: %s\n", opts.mode);
    printf("Threads (P): %d\n", p_value);
    printf("Block size: %d\n", opts.block_size);
    printf("Tempo: %.6f s\n", elapsed);
    printf("FLOPS: %.2f (%.2f MFLOPS)\n", flops, mflops);
    if (metric_is_na(speedup)) printf("Speedup: N/A\n");
    else printf("Speedup: %.2fx\n", speedup);
    if (metric_is_na(efficiency)) printf("Eficiência: N/A\n");
    else printf("Eficiência: %.2f\n", efficiency);
    if (metric_is_na(energy_j)) printf("Energia (RAPL): N/A\n");
    else printf("Energia: %.4f J (%.2f W)\n", energy_j, avg_power);
    if (metric_is_na(mflops_per_watt)) printf("MFLOPS/Watt: N/A\n");
    else printf("MFLOPS/Watt: %.4f\n", mflops_per_watt);
    printf("Saída: %s\n", final_output_path);

    // Salvar métricas em JSON se especificado
    if (opts.metrics_path) {
        // Forçar caminho relativo à pasta results/
        char final_metrics_path[1024];
        if (strncmp(opts.metrics_path, "results/", 8) != 0 && strncmp(opts.metrics_path, "tests/", 6) != 0) {
            snprintf(final_metrics_path, sizeof(final_metrics_path), "results/%s", opts.metrics_path);
        } else {
            strncpy(final_metrics_path, opts.metrics_path, sizeof(final_metrics_path));
        }
        // Criar diretório de métricas se não existir
        char *last_slash = strrchr(final_metrics_path, '/');
        if (last_slash) {
            char dir[512];
            size_t dir_len = last_slash - final_metrics_path;
            strncpy(dir, final_metrics_path, dir_len);
            dir[dir_len] = '\0';
            mkdir(dir, 0755);
        }
        FILE *f = fopen(final_metrics_path, "w");
        if (f) {
            fprintf(f, "{\n");
            fprintf(f, "  \"image\": \"%s\",\n", opts.image_path);
            fprintf(f, "  \"image_dims\": [%d, %d],\n", img.width, img.height);
            fprintf(f, "  \"kernel\": \"%s\",\n", opts.kernel_name);
            fprintf(f, "  \"kernel_size\": %d,\n", opts.kernel_size);
            fprintf(f, "  \"mode\": \"%s\",\n", opts.mode);
            fprintf(f, "  \"threads_p\": %d,\n", p_value);
            fprintf(f, "  \"block_size\": %d,\n", opts.block_size);
            fprintf(f, "  \"time_seconds\": %.6f,\n", elapsed);
            fprintf(f, "  \"flops\": %.2f,\n", flops);
            fprintf(f, "  \"mflops\": %.4f,\n", mflops);
            /* Campos que podem ser N/A são escritos como null em JSON. */
            if (metric_is_na(speedup)) fprintf(f, "  \"speedup\": null,\n");
            else fprintf(f, "  \"speedup\": %.4f,\n", speedup);
            if (metric_is_na(efficiency)) fprintf(f, "  \"efficiency\": null,\n");
            else fprintf(f, "  \"efficiency\": %.4f,\n", efficiency);
            if (metric_is_na(energy_j)) fprintf(f, "  \"energy_joules\": null,\n");
            else fprintf(f, "  \"energy_joules\": %.6f,\n", energy_j);
            if (metric_is_na(avg_power)) fprintf(f, "  \"avg_power_w\": null,\n");
            else fprintf(f, "  \"avg_power_w\": %.4f,\n", avg_power);
            if (metric_is_na(mflops_per_watt)) fprintf(f, "  \"mflops_per_watt\": null\n");
            else fprintf(f, "  \"mflops_per_watt\": %.6f\n", mflops_per_watt);
            fprintf(f, "}\n");
            fclose(f);
            printf("Métricas salvas em: %s\n", final_metrics_path);
        } else {
            fprintf(stderr, "Aviso: não foi possível salvar métricas em %s\n", final_metrics_path);
        }
    }

    // Limpeza
    free_image(&img);
    free_image(&out_img);

    return 0;
}
