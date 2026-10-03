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

    #define NUM_KERNELS 5
    #define NUM_MODES 4
    #define NUM_RUNS 10

    // Set default values for options
    opts.image_path = '../../images/a.jpg';
    opts.kernel_name = NULL;
    opts.mode = DEFAULT_MODE;
    opts.block_size = DEFAULT_BLOCK_SIZE;
    opts.kernel_size = DEFAULT_KERNEL_SIZE;
    opts.threads = DEFAULT_THREADS;
    opts.output_path = DEFAULT_OUTPUT;
    opts.metrics_path = DEFAULT_METRICS;

    Image img = load_image(opts.image_path);
    if (!img.data) {
        fprintf(stderr, "Erro ao carregar a imagem: %s\n", opts.image_path);
        return 1;
    }

    Kernel filters[5];
    get_convolution_filters(filters);

    for (int i = 0; i < 5; i++) {
        filters[i].size = opts.kernel_size;
    }

    Image out_img;
    out_img.width = img.width;
    out_img.height = img.height;
    out_img.data = NULL;

    convolution_set_num_threads(opts.threads);

    double times[NUM_KERNELS][NUM_MODES] = {{0.0}};
    double energy_totals[NUM_KERNELS][NUM_MODES] = {{0.0}};
    int energy_counts[NUM_KERNELS][NUM_MODES] = {{0}};

    // Start processing the image with all kernels and modes
    for (int kernel_index = 0; kernel_index < NUM_KERNELS; kernel_index++) {
        opts.kernel_name = kernel_names[kernel_index];
        for (int run_index = 0; run_index < NUM_RUNS; run_index++) {
            for (int mode_index = 0; mode_index < NUM_MODES; mode_index++) {
                opts.mode = mode_names[mode_index];

                // Prepare output path
                char output_path[1024];
                snprintf(output_path, sizeof(output_path), "results/%s_%s_run%d.png", opts.kernel_name, opts.mode, run_index + 1);
                opts.output_path = output_path;
                out_img.data = NULL;

                EnergySample energy_begin = energy_read();
                Timer timer;
                timer_start(&timer);

                switch (mode_index) {
                case 0: // Sequential
                    apply_convolution_sequential(&img, &filters[kernel_index], &out_img);
                    break;
                case 1: // OpenMP
                    apply_convolution_openmp(&img, &filters[kernel_index], &out_img);
                    break;
                case 2: // Pthreads
                    apply_convolution_pthread(&img, &filters[kernel_index], &out_img);
                    break;
                case 3: // CUDA
                    apply_convolution_cuda(&img, &filters[kernel_index], &out_img);
                    break;
                    break;
                }
                
                double elapsed_time = timer_stop(&timer);
                EnergySample energy_end = energy_read();
                double energy_joules = energy_delta_joules(&energy_begin, &energy_end);

                times[kernel_index][mode_index] += elapsed_time;
                if (energy_joules >= 0.0) {
                    energy_totals[kernel_index][mode_index] += energy_joules;
                    energy_counts[kernel_index][mode_index]++;
                }

                save_image(opts.output_path, &out_img);
                free_image(&out_img);
            }
        }
    }

    printf("=== Resultados médios (%d execuções) ===\n", NUM_RUNS);
    for (int kernel_index = 0; kernel_index < NUM_KERNELS; kernel_index++) {
        double sequential_average = times[kernel_index][0] / NUM_RUNS;

        printf("\nKernel: %s\n", kernel_names[kernel_index]);
        for (int mode_index = 0; mode_index < NUM_MODES; mode_index++) {
            double average_time = times[kernel_index][mode_index] / NUM_RUNS;
            double average_energy = METRIC_NA;
            if (energy_counts[kernel_index][mode_index] > 0) {
                average_energy = energy_totals[kernel_index][mode_index] /
                                 energy_counts[kernel_index][mode_index];
            }

            double speedup = metrics_speedup(sequential_average, average_time);
            if (mode_index == 0) {
                speedup = 1.0;
            }

            printf("  %-7s | tempo médio: %.6f s | speedup: ",
                   mode_names[mode_index], average_time);
            if (metric_is_na(speedup)) {
                printf("N/A");
            } else {
                printf("%.2fx", speedup);
            }
            printf(" | energia média: ");
            if (metric_is_na(average_energy)) {
                printf("N/A\n");
            } else {
                printf("%.6f J\n", average_energy);
            }
        }
    }

    free_image(&img);
    return 0;
}