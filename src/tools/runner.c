/* runner.c — cycle-test driver.
 *
 * Um "cycle test" executa UM algoritmo (seq | omp | pthread | cuda) N vezes
 * sobre uma imagem e um kernel escolhidos, recolhendo métricas por execução
 * (tempo, FLOPS/MFLOPS, energia via RAPL, potência, MFLOPS/Watt, speedup,
 * eficiência) e gravando as N linhas num ficheiro CSV.
 *
 * P (número de processadores) = núcleos físicos/on-line da máquina
 * (metrics_cpu_cores()). Speedup(P) = T_s / T_p, onde T_s é a média de um
 * ciclo sequencial de baseline sobre a mesma imagem+kernel. Eficiência =
 * Speedup / P (N/A para CUDA, onde P não corresponde a núcleos de CPU).
 *
 * Uso:
 *   runner -m <modo> -i <imagem> -k <kernel> -n <reps> [-c <csv>] [-s <size>]
 */

#include "image.h"
#include "kernel.h"
#include "convolution.h"
#include "timer.h"
#include "metrics.h"
#include "energy.h"
#include "csv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#define DEFAULT_REPS 10
#define DEFAULT_KERNEL_SIZE 3
#define DEFAULT_THREADS 4

static const char *kernel_names[] = {
    "laplacian", "sobel_h", "sobel_v", "sharpen", "blur"
};
static const char *mode_names[] = { "seq", "omp", "pthread", "cuda" };

typedef struct {
    const char *mode;
    const char *image;
    const char *kernel;
    int reps;
    int kernel_size;
    int threads;
    const char *csv_path;
} RunnerOpts;

static int index_of(const char *name, const char **arr, int n) {
    for (int i = 0; i < n; i++) {
        if (strcmp(name, arr[i]) == 0) return i;
    }
    return -1;
}

static void print_usage(const char *prog) {
    printf("Uso: %s -m <modo> -i <imagem> -k <kernel> -n <reps> [-c <csv>] [-s <size>] [-p <threads>]\n", prog);
    printf("\nOpções:\n");
    printf("  -m  Modo: seq, omp, pthread, cuda\n");
    printf("  -i  Caminho da imagem de entrada (obrigatório)\n");
    printf("  -k  Kernel: laplacian, sobel_h, sobel_v, sharpen, blur (obrigatório)\n");
    printf("  -n  Número de execuções no ciclo (padrão: %d)\n", DEFAULT_REPS);
    printf("  -c  Caminho do CSV de saída (padrão: results/csv/<modo>_<kernel>.csv)\n");
    printf("  -s  Tamanho do kernel (padrão: %d)\n", DEFAULT_KERNEL_SIZE);
    printf("  -p  Número de threads (omp/pthread; padrão: %d)\n", DEFAULT_THREADS);
    printf("  -h  Ajuda\n");
}

/* Executa a convolução do modo indicado, alocando output->data internamente. */
static void run_convolution(int mode_idx, Image *in, Kernel *k, Image *out) {
    switch (mode_idx) {
        case 0: apply_convolution_sequential(in, k, out); break;
        case 1: apply_convolution_openmp(in, k, out); break;
        case 2: apply_convolution_pthread(in, k, out); break;
        case 3: apply_convolution_cuda(in, k, out); break;
        default: break;
    }
}

/* Executa uma única iteração cronometrada + medição de energia.
 * Devolve o tempo (s); preenche *energy_j (J ou METRIC_NA). */
static double timed_run(int mode_idx, Image *in, Kernel *k, double *energy_j) {
    Image out;
    out.width = in->width;
    out.height = in->height;
    out.data = NULL;

    EnergySample eb = energy_read();
    Timer t;
    timer_start(&t);
    run_convolution(mode_idx, in, k, &out);
    double elapsed = timer_stop(&t);
    EnergySample ee = energy_read();

    double ej = energy_delta_joules(&eb, &ee);
    *energy_j = (ej < 0.0) ? METRIC_NA : ej;

    free_image(&out);
    return elapsed;
}

int main(int argc, char **argv) {
    RunnerOpts opts = {
        .mode = "seq", .image = NULL, .kernel = NULL,
        .reps = DEFAULT_REPS, .kernel_size = DEFAULT_KERNEL_SIZE,
        .threads = DEFAULT_THREADS, .csv_path = NULL
    };

    int opt;
    while ((opt = getopt(argc, argv, "m:i:k:n:c:s:p:h")) != -1) {
        switch (opt) {
            case 'm': opts.mode = optarg; break;
            case 'i': opts.image = optarg; break;
            case 'k': opts.kernel = optarg; break;
            case 'n': opts.reps = atoi(optarg); break;
            case 'c': opts.csv_path = optarg; break;
            case 's': opts.kernel_size = atoi(optarg); break;
            case 'p': opts.threads = atoi(optarg); break;
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (!opts.image || !opts.kernel) {
        fprintf(stderr, "Erro: -i (imagem) e -k (kernel) são obrigatórios\n");
        print_usage(argv[0]);
        return 1;
    }
    if (opts.reps <= 0) {
        fprintf(stderr, "Erro: -n deve ser positivo\n");
        return 1;
    }
    if (opts.kernel_size <= 0 || opts.kernel_size % 2 == 0) {
        fprintf(stderr, "Erro: tamanho do kernel deve ser ímpar e positivo\n");
        return 1;
    }
    if (opts.threads < 1) {
        fprintf(stderr, "Erro: -p (threads) deve ser >= 1\n");
        return 1;
    }

    int mode_idx = index_of(opts.mode, mode_names, 4);
    if (mode_idx < 0) {
        fprintf(stderr, "Erro: modo '%s' desconhecido\n", opts.mode);
        return 1;
    }
    int kernel_idx = index_of(opts.kernel, kernel_names, 5);
    if (kernel_idx < 0) {
        fprintf(stderr, "Erro: kernel '%s' desconhecido\n", opts.kernel);
        return 1;
    }

    /* Caminho CSV padrão: results/csv/<modo>_<kernel>.csv */
    char csv_buf[1024];
    const char *csv_path = opts.csv_path;
    if (!csv_path) {
        snprintf(csv_buf, sizeof(csv_buf), "results/csv/%s_%s.csv",
                 opts.mode, opts.kernel);
        csv_path = csv_buf;
    }

    /* Carrega imagem uma vez. */
    Image img = load_image(opts.image);
    if (!img.data) {
        fprintf(stderr, "Erro ao carregar a imagem: %s\n", opts.image);
        return 1;
    }

    /* Kernels. */
    Kernel filters[5];
    get_convolution_filters(filters);
    for (int i = 0; i < 5; i++) filters[i].size = opts.kernel_size;
    Kernel *k = &filters[kernel_idx];

    /* Configura o número de threads para os backends omp/pthread. */
    convolution_set_num_threads(opts.threads);

    /* P = número de threads: seq usa 1, omp/pthread usam o valor de -p.
     * (CUDA não usa P para eficiência — ver abaixo.) */
    int p_value = (mode_idx == 0) ? 1 : opts.threads;

    /* Baseline sequencial (média de um ciclo) para T_s, exceto no modo seq. */
    double t_s_mean = METRIC_NA;
    if (mode_idx != 0) {
        double sum = 0.0;
        for (int i = 0; i < opts.reps; i++) {
            double ej;
            sum += timed_run(0, &img, k, &ej);
        }
        t_s_mean = sum / opts.reps;
        printf("Baseline sequencial (média de %d execuções): T_s = %.6f s\n",
               opts.reps, t_s_mean);
    }

    /* Abre CSV. */
    FILE *csv = csv_open(csv_path);
    if (!csv) {
        fprintf(stderr, "Erro: não foi possível abrir o CSV %s\n", csv_path);
        free_image(&img);
        return 1;
    }

    printf("=== Cycle test ===\n");
    printf("Modo: %s | Imagem: %s (%dx%d) | Kernel: %s (size=%d) | N=%d | threads(P)=%d\n",
           opts.mode, opts.image, img.width, img.height,
           opts.kernel, opts.kernel_size, opts.reps, p_value);

    /* Acumuladores para o resumo. */
    double sum_time = 0.0, min_time = 1e30, max_time = 0.0;
    double sum_speedup = 0.0, sum_eff = 0.0, sum_mpw = 0.0;
    int speedup_n = 0, eff_n = 0, mpw_n = 0;

    for (int i = 1; i <= opts.reps; i++) {
        double energy_j;
        double t_p = timed_run(mode_idx, &img, k, &energy_j);

        double flops = metrics_flops(img.width, img.height, 3, opts.kernel_size, t_p);
        double mflops = metrics_mflops(flops);
        double avg_power = metrics_avg_power(energy_j, t_p);
        double mflops_per_watt = metrics_mflops_per_watt(mflops, avg_power);

        double speedup;
        if (mode_idx == 0) {
            speedup = 1.0; /* seq comparado consigo próprio */
        } else {
            speedup = metrics_speedup(t_s_mean, t_p);
        }

        double efficiency;
        if (mode_idx == 3) {
            efficiency = METRIC_NA; /* CUDA: P não corresponde a threads de CPU */
        } else {
            efficiency = metrics_efficiency(speedup, p_value);
        }

        Metrics m = {
            .mode = opts.mode, .image = opts.image,
            .image_w = img.width, .image_h = img.height,
            .kernel = opts.kernel, .kernel_size = opts.kernel_size,
            .run_index = i, .threads_p = p_value,
            .time_s = t_p, .flops = flops, .mflops = mflops,
            .energy_joules = energy_j, .avg_power_w = avg_power,
            .mflops_per_watt = mflops_per_watt,
            .speedup = speedup, .efficiency = efficiency
        };
        csv_append_row(csv, &m);

        sum_time += t_p;
        if (t_p < min_time) min_time = t_p;
        if (t_p > max_time) max_time = t_p;
        if (!metric_is_na(speedup)) { sum_speedup += speedup; speedup_n++; }
        if (!metric_is_na(efficiency)) { sum_eff += efficiency; eff_n++; }
        if (!metric_is_na(mflops_per_watt)) { sum_mpw += mflops_per_watt; mpw_n++; }

        printf("  run %2d/%d: t=%.6f s, %.1f MFLOPS", i, opts.reps, t_p, mflops);
        if (!metric_is_na(speedup)) printf(", speedup=%.2f", speedup);
        if (!metric_is_na(mflops_per_watt)) printf(", %.2f MFLOPS/W", mflops_per_watt);
        printf("\n");
    }

    csv_close(csv);

    printf("--- Resumo do ciclo ---\n");
    printf("Tempo: média=%.6f s, min=%.6f s, max=%.6f s\n",
           sum_time / opts.reps, min_time, max_time);
    if (speedup_n > 0) printf("Speedup médio: %.2f\n", sum_speedup / speedup_n);
    else printf("Speedup médio: N/A\n");
    if (eff_n > 0) printf("Eficiência média: %.2f\n", sum_eff / eff_n);
    else printf("Eficiência média: N/A\n");
    if (mpw_n > 0) printf("MFLOPS/Watt médio: %.4f\n", sum_mpw / mpw_n);
    else printf("MFLOPS/Watt médio: N/A (RAPL indisponível)\n");
    printf("CSV: %s\n", csv_path);

    free_image(&img);
    return 0;
}
