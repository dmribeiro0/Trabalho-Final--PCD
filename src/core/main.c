#include "image.h"
#include "kernel.h"
#include "convolution.h"
#include "timer.h"
#include "metrics.h"
#include "energy.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// Mapa de kernels
static const char *kernel_names[] = {
    "laplacian", "sobel_h", "sobel_v", "sharpen", "blur"
};

static const char *mode_names[] = {
    "sequential", "openmp", "pthread"
};

typedef struct {
    double time_elapsed;
    double energy_consumed;
    double speedup;
    double efficiency;
    double flops;
    double mflops;
    double avg_power;
    double mflops_per_watt;
} RunStats;

typedef struct {
    RunStats runs[10];
    double avg_time;
    double avg_energy;
    double avg_speedup;
    double avg_efficiency;
    double avg_flops;
    double avg_mflops;
    double avg_power;
    double avg_mflops_per_watt;
} KernelStats;

typedef struct {
    const char *mode;
    KernelStats kernel_stats[5];
} ModeStats;

KernelStats create_kernel_stats() {
    KernelStats stats;
    for (int i = 0; i < 10; i++) {
        stats.runs[i].time_elapsed = 0.0;
        stats.runs[i].energy_consumed = 0.0;
        stats.runs[i].speedup = 0.0;
        stats.runs[i].efficiency = 0.0;
        stats.runs[i].flops = 0.0;
        stats.runs[i].mflops = 0.0;
        stats.runs[i].avg_power = 0.0;
        stats.runs[i].mflops_per_watt = 0.0;
    }
    stats.avg_time = 0.0;
    stats.avg_energy = 0.0;
    stats.avg_speedup = 0.0;
    stats.avg_efficiency = 0.0;
    stats.avg_flops = 0.0;
    stats.avg_mflops = 0.0;
    stats.avg_power = 0.0;
    stats.avg_mflops_per_watt = 0.0;

    return stats;
}

ModeStats create_mode_stats(const char *mode) {
    ModeStats stats;
    stats.mode = mode;
    for (int i = 0; i < 5; i++) {
        stats.kernel_stats[i] = create_kernel_stats();
    }
    return stats;
}

void calculate_run_stats(KernelStats *kernel_stats) {
    double total_time = 0.0;
    double total_energy = 0.0;
    double total_speedup = 0.0;
    double total_efficiency = 0.0;
    double total_flops = 0.0;
    double total_mflops = 0.0;
    double total_power = 0.0;
    double total_mflops_per_watt = 0.0;

    for (int i = 0; i < 10; i++) {
        total_time += kernel_stats->runs[i].time_elapsed;
        total_energy += kernel_stats->runs[i].energy_consumed;
        total_speedup += kernel_stats->runs[i].speedup;
        total_efficiency += kernel_stats->runs[i].efficiency;
        total_flops += kernel_stats->runs[i].flops;
        total_mflops += kernel_stats->runs[i].mflops;
        total_power += kernel_stats->runs[i].avg_power;
        total_mflops_per_watt += kernel_stats->runs[i].mflops_per_watt;
    }

    kernel_stats->avg_time = total_time / 10.0;
    kernel_stats->avg_energy = total_energy / 10.0;
    kernel_stats->avg_speedup = total_speedup / 10.0;
    kernel_stats->avg_efficiency = total_efficiency / 10.0;
    kernel_stats->avg_flops = total_flops / 10.0;
    kernel_stats->avg_mflops = total_mflops / 10.0;
    kernel_stats->avg_power = total_power / 10.0;
    kernel_stats->avg_mflops_per_watt = total_mflops_per_watt / 10.0;
}

void save_kernel_stats(const char *filename, const KernelStats *kernel_stats) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar as estatísticas: %s\n", filename);
        return;
    }

    fprintf(file, "Run,Time (s),Energy (J),Speedup,Efficiency,Flops,MFlops,Avg Power (W),MFlops/W\n");
    for (int i = 0; i < 10; i++) {
        fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                i + 1,
                kernel_stats->runs[i].time_elapsed,
                kernel_stats->runs[i].energy_consumed,
                kernel_stats->runs[i].speedup,
                kernel_stats->runs[i].efficiency,
                kernel_stats->runs[i].flops,
                kernel_stats->runs[i].mflops,
                kernel_stats->runs[i].avg_power,
                kernel_stats->runs[i].mflops_per_watt);
    }
    fclose(file);
}

void save_average_stats(const char *filename, const KernelStats *kernel_stats) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar as estatísticas médias: %s\n", filename);
        return;
    }

    fprintf(file, "Average Time (s),Average Energy (J),Average Speedup,Average Efficiency,Average Flops,Average MFlops,Average Power (W),Average MFlops/W\n");
    fprintf(file, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            kernel_stats->avg_time,
            kernel_stats->avg_energy,
            kernel_stats->avg_speedup,
            kernel_stats->avg_efficiency,
            kernel_stats->avg_flops,
            kernel_stats->avg_mflops,
            kernel_stats->avg_power,
            kernel_stats->avg_mflops_per_watt);

    fclose(file);
}

int main(void) {
    const char *image_path = "images/a.jpg";
    const int kernel_size = 3;
    const int threads = 4;
    const int num_runs = 10;
    int mode_index = 0;

    Image img = load_image(image_path);
    if (!img.data) {
        fprintf(stderr, "Erro ao carregar a imagem: %s\n", image_path);
        return 1;
    }

    Kernel filters[5];
    get_convolution_filters(filters);
    for (int i = 0; i < 5; i++) {
        filters[i].size = kernel_size;
    }

    convolution_set_num_threads(threads);
    mkdir("outputs", 0755);
    mkdir("outputs/sequential", 0755);
    mkdir("outputs/sequential/images", 0755);
    mkdir("outputs/sequential/csv", 0755);
    mkdir("outputs/openmp", 0755);
    mkdir("outputs/openmp/images", 0755);
    mkdir("outputs/openmp/csv", 0755);
    mkdir("outputs/pthread", 0755);
    mkdir("outputs/pthread/images", 0755);
    mkdir("outputs/pthread/csv", 0755);


    ModeStats mode_stats[3];
    
    for (int i = 0; i < 3; i++) {
        mode_stats[i] = create_mode_stats(mode_names[i]);
    }

    for (mode_index = 0; mode_index < 3; mode_index++) {
        const char *current_mode = mode_names[mode_index];
        printf("=== Experimento: %d execucoes por kernel (%s) ===\n", num_runs, current_mode);
        for (int kernel_index = 0; kernel_index < 5; kernel_index++) {
            KernelStats *current_stats = &mode_stats[mode_index].kernel_stats[kernel_index];

            printf("\nKernel: %s\n", kernel_names[kernel_index]);
            for (int run_index = 0; run_index < num_runs; run_index++) {
                Image out_img = {img.width, img.height, NULL};
                EnergySample energy_begin = energy_read();
                Timer timer;
                timer_start(&timer);

                switch (mode_index) {
                case 0:
                    apply_convolution_sequential(&img, &filters[kernel_index], &out_img);
                    break;
                case 1:
                    apply_convolution_openmp(&img, &filters[kernel_index], &out_img);
                    break;
                case 2:
                    apply_convolution_pthread(&img, &filters[kernel_index], &out_img);
                    break;
                }

                current_stats->runs[run_index].time_elapsed = timer_stop(&timer);
                EnergySample energy_end = energy_read();
                current_stats->runs[run_index].energy_consumed = energy_delta_joules(&energy_begin, &energy_end);
                double sequential_time = mode_stats[0].kernel_stats[kernel_index].avg_time;
                int processors = (mode_index == 0) ? 1 : threads;
                current_stats->runs[run_index].speedup =
                    (mode_index == 0)
                        ? 1.0
                        : metrics_speedup(sequential_time,
                                          current_stats->runs[run_index].time_elapsed);
                current_stats->runs[run_index].efficiency =
                    metrics_efficiency(current_stats->runs[run_index].speedup, processors);
                current_stats->runs[run_index].flops = metrics_flops(img.width, img.height, 3, kernel_size, current_stats->runs[run_index].time_elapsed);
                current_stats->runs[run_index].mflops = metrics_mflops(current_stats->runs[run_index].flops);
                current_stats->runs[run_index].avg_power = metrics_avg_power(current_stats->runs[run_index].energy_consumed, current_stats->runs[run_index].time_elapsed);
                current_stats->runs[run_index].mflops_per_watt = metrics_mflops_per_watt(current_stats->runs[run_index].mflops, current_stats->runs[run_index].avg_power);

                char output_path[256];
                snprintf(output_path, sizeof(output_path),
                        "outputs/%s/images/%s_%s_run%d.png", current_mode, kernel_names[kernel_index], current_mode, run_index + 1);
                save_image(output_path, &out_img);

                free_image(&out_img);
            }

            calculate_run_stats(current_stats);

            /* Imprime as estatísticas médias */
            /*
            printf("=== Estatísticas médias para o kernel %s (%s) ===\n", kernel_names[kernel_index], current_mode);
            printf("Tempo médio: %.6f s\n", current_stats->avg_time);
            printf("Energia média: %.6f J\n", current_stats->avg_energy);
            printf("Speedup médio: %.6f\n", current_stats->avg_speedup);
            printf("Eficiência média: %.6f\n", current_stats->avg_efficiency);
            printf("FLOPS médio: %.6f\n", current_stats->avg_flops);
            printf("MFLOPS médio: %.6f\n", current_stats->avg_mflops);
            printf("Potência média: %.6f W\n", current_stats->avg_power);
            printf("MFLOPS/W médio: %.6f\n", current_stats->avg_mflops_per_watt);
            */

            char stats_path[256];
            snprintf(stats_path, sizeof(stats_path),
                     "outputs/%s/csv/%s_%s_stats.csv",
                     current_mode, kernel_names[kernel_index], current_mode);
            save_kernel_stats(stats_path, current_stats);

            snprintf(stats_path, sizeof(stats_path),
                     "outputs/%s/csv/%s_%s_avg_stats.csv",
                     current_mode, kernel_names[kernel_index], current_mode);
            save_average_stats(stats_path, current_stats);
        }
    }
    

    free_image(&img);
    return 0;
}