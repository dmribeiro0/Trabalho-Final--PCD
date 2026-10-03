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

int main(void) {
    const char *image_path = "images/a.jpg";
    const char *mode = "seq";
    const int kernel_size = 3;
    const int threads = 4;
    const int num_runs = 10;
    const int mode_index = 0;

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

    KernelStats kernel_stats = create_kernel_stats();

    printf("=== Experimento: %d execucoes por kernel (%s) ===\n", num_runs, mode);
    for (int kernel_index = 0; kernel_index < 5; kernel_index++) {
        double total_time = 0.0;
        double total_energy = 0.0;
        int energy_count = 0;

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
            case 3:
                apply_convolution_cuda(&img, &filters[kernel_index], &out_img);
                break;
            }

            kernel_stats.runs[run_index].time_elapsed = timer_stop(&timer);
            EnergySample energy_end = energy_read();
            kernel_stats.runs[run_index].energy_consumed = energy_delta_joules(&energy_begin, &energy_end);
            kernel_stats.runs[run_index].speedup = 1.0;
            kernel_stats.runs[run_index].efficiency = metrics_efficiency(kernel_stats.runs[run_index].speedup, 1);
            kernel_stats.runs[run_index].flops = metrics_flops(img.width, img.height, 3, kernel_size, kernel_stats.runs[run_index].time_elapsed);
            kernel_stats.runs[run_index].mflops = metrics_mflops(kernel_stats.runs[run_index].flops);
            kernel_stats.runs[run_index].avg_power = metrics_avg_power(kernel_stats.runs[run_index].energy_consumed, kernel_stats.runs[run_index].time_elapsed);
            kernel_stats.runs[run_index].mflops_per_watt = metrics_mflops_per_watt(kernel_stats.runs[run_index].mflops, kernel_stats.runs[run_index].avg_power);

            char output_path[256];
            snprintf(output_path, sizeof(output_path),
                     "outputs/%s_%s_run%d.png", kernel_names[kernel_index], mode, run_index + 1);
            save_image(output_path, &out_img);

            total_time += elapsed;
            if (!metric_is_na(energy)) {
                total_energy += energy;
                energy_count++;
            }
                 printf("  execucao %d: %.6f s | %.2f FLOPS | %.2f MFLOPS | ",
                     run_index + 1, elapsed, flops, mflops);
                 if (metric_is_na(energy)) printf("energia: N/A | ");
                 else printf("energia: %.6f J | ", energy);
                 if (metric_is_na(avg_power)) printf("potencia: N/A | ");
                 else printf("potencia: %.6f W | ", avg_power);
                 if (metric_is_na(mflops_per_watt)) printf("MFLOPS/W: N/A | ");
                 else printf("MFLOPS/W: %.6f | ", mflops_per_watt);
                 printf("speedup: %.2fx | eficiencia: %.2f | imagem: %s\n",
                     speedup, efficiency, output_path);
            free_image(&out_img);
        }

        printf("  media: %.6f s | speedup: 1.00x | eficiencia: 1.00\n",
               total_time / num_runs);
        if (energy_count > 0) {
            printf("  energia media: %.6f J\n", total_energy / energy_count);
        } else {
            printf("  energia media: N/A\n");
        }
    }

    free_image(&img);
    return 0;
}