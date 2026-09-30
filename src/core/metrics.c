/* metrics.c — metrics module implementation.
 *
 * Implements the interface declared in include/core/metrics.h: derived
 * performance metrics (FLOPS, MFLOPS, speedup, efficiency, MFLOPS/Watt)
 * built on top of raw timing and energy readings.
 */
#include "metrics.h"

#include <unistd.h>

double metrics_total_flops(int width, int height, int channels, int k) {
    if (width <= 0 || height <= 0 || channels <= 0 || k <= 0) {
        return 0.0;
    }
    /* k*k multiplicações + k*k somas por canal por pixel = 2*k*k FLOPs. */
    return 2.0 * (double)width * (double)height * (double)channels *
           (double)k * (double)k;
}

double metrics_flops(int width, int height, int channels, int k, double time_s) {
    if (time_s <= 0.0) {
        return 0.0;
    }
    return metrics_total_flops(width, height, channels, k) / time_s;
}

double metrics_mflops(double flops) {
    return flops / 1e6;
}

double metrics_speedup(double t_s, double t_p) {
    if (t_s <= 0.0 || t_p <= 0.0) {
        return METRIC_NA;
    }
    return t_s / t_p;
}

double metrics_efficiency(double speedup, int p) {
    if (metric_is_na(speedup) || p <= 0) {
        return METRIC_NA;
    }
    return speedup / (double)p;
}

double metrics_avg_power(double energy_joules, double time_s) {
    if (metric_is_na(energy_joules) || time_s <= 0.0) {
        return METRIC_NA;
    }
    return energy_joules / time_s;
}

double metrics_mflops_per_watt(double mflops, double avg_power_w) {
    if (metric_is_na(avg_power_w) || avg_power_w <= 0.0) {
        return METRIC_NA;
    }
    return mflops / avg_power_w;
}

int metrics_cpu_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) {
        return 1;
    }
    return (int)n;
}

bool metric_is_na(double v) {
    return v <= METRIC_NA;
}
