/* metrics.h — performance metrics interface.
 *
 * Public interface for the metrics module: derived performance metrics
 * (FLOPS, MFLOPS, speedup, efficiency, MFLOPS/Watt) computed from elapsed
 * times (see timer.h), operation counts and energy readings (see energy.h).
 * Used to compare the sequential baseline against the OpenMP / Pthreads /
 * CUDA versions.
 *
 * Definições:
 *   Speedup(P)   = T_s / T_p(P)
 *   Efficiency   = Speedup(P) / P   (P = número de threads; seq=1, CUDA=N/A)
 *   FLOPS        = total_flops / tempo
 *   MFLOPS/Watt  = MFLOPS / potência_média
 *
 * Contagem de FLOPs (frame completo): cada pixel de saída faz k*k
 * multiplicações e k*k somas por canal, portanto
 *   total_flops = 2 * width * height * channels * k * k.
 */
#ifndef METRICS_H
#define METRICS_H

#include <stdbool.h>

/* Valor sentinela usado quando uma métrica não está disponível
 * (ex.: energia sem RAPL, ou eficiência para CUDA). */
#define METRIC_NA (-1.0)

/* Registro completo de uma única execução (uma linha do CSV). */
typedef struct {
    const char *mode;        /* seq | omp | pthread | cuda */
    const char *image;       /* caminho da imagem de entrada */
    int         image_w;
    int         image_h;
    const char *kernel;      /* nome do filtro */
    int         kernel_size; /* k */
    int         run_index;   /* índice da execução no ciclo (1..N) */
    int         threads_p;   /* P — número de threads (omp/pthread); seq=1 */

    double      time_s;              /* tempo da execução (s) */
    double      flops;               /* operações de ponto flutuante por segundo */
    double      mflops;              /* flops / 1e6 */
    double      energy_joules;       /* energia consumida (J) ou METRIC_NA */
    double      avg_power_w;         /* potência média (W) ou METRIC_NA */
    double      mflops_per_watt;     /* MFLOPS / W ou METRIC_NA */
    double      speedup;             /* T_s / T_p ou METRIC_NA */
    double      efficiency;          /* speedup / P ou METRIC_NA */
} Metrics;

/* Número total de FLOPs para uma convolução de frame completo. */
double metrics_total_flops(int width, int height, int channels, int k);

/* FLOPS = total_flops / tempo. Retorna 0 se tempo <= 0. */
double metrics_flops(int width, int height, int channels, int k, double time_s);

/* MFLOPS = FLOPS / 1e6. */
double metrics_mflops(double flops);

/* Speedup(P) = T_s / T_p. Retorna METRIC_NA se algum tempo for inválido. */
double metrics_speedup(double t_s, double t_p);

/* Efficiency = Speedup / P. Retorna METRIC_NA se entrada inválida. */
double metrics_efficiency(double speedup, int p);

/* Potência média (W) = energia (J) / tempo (s). METRIC_NA se indisponível. */
double metrics_avg_power(double energy_joules, double time_s);

/* MFLOPS/Watt. METRIC_NA se potência indisponível ou <= 0. */
double metrics_mflops_per_watt(double mflops, double avg_power_w);

/* Número de núcleos de CPU on-line (P). Mínimo 1. */
int metrics_cpu_cores(void);

/* Verdadeiro se o valor é o sentinela "não disponível". */
bool metric_is_na(double v);

#endif /* METRICS_H */
