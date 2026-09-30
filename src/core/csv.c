/* csv.c — CSV writer implementation for cycle-test metrics. */
#define _POSIX_C_SOURCE 200809L
#include "csv.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define CSV_HEADER \
    "timestamp,mode,image,image_w,image_h,kernel,kernel_size,run_index," \
    "threads_p,time_s,flops,mflops,energy_joules,avg_power_w,mflops_per_watt," \
    "speedup,efficiency\n"

/* Cria recursivamente os diretórios pais de 'path'. */
static void make_parent_dirs(const char *path) {
    char buf[1024];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0755);
            *p = '/';
        }
    }
}

/* Escreve um valor double ou "NA" se for o sentinela. */
static void write_metric(FILE *f, double v) {
    if (metric_is_na(v)) {
        fprintf(f, "NA");
    } else {
        fprintf(f, "%.6f", v);
    }
}

FILE *csv_open(const char *path) {
    make_parent_dirs(path);

    /* Verifica se o ficheiro já tem conteúdo (para decidir sobre o cabeçalho). */
    int need_header = 1;
    FILE *probe = fopen(path, "r");
    if (probe) {
        if (fgetc(probe) != EOF) {
            need_header = 0;
        }
        fclose(probe);
    }

    FILE *f = fopen(path, "a");
    if (!f) {
        return NULL;
    }
    if (need_header) {
        fputs(CSV_HEADER, f);
    }
    return f;
}

void csv_append_row(FILE *f, const Metrics *m) {
    if (!f || !m) {
        return;
    }

    time_t now = time(NULL);
    char ts[32];
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tmv);

    fprintf(f, "%s,%s,%s,%d,%d,%s,%d,%d,%d,",
            ts,
            m->mode ? m->mode : "",
            m->image ? m->image : "",
            m->image_w, m->image_h,
            m->kernel ? m->kernel : "",
            m->kernel_size,
            m->run_index,
            m->threads_p);

    write_metric(f, m->time_s);           fputc(',', f);
    write_metric(f, m->flops);            fputc(',', f);
    write_metric(f, m->mflops);           fputc(',', f);
    write_metric(f, m->energy_joules);    fputc(',', f);
    write_metric(f, m->avg_power_w);      fputc(',', f);
    write_metric(f, m->mflops_per_watt);  fputc(',', f);
    write_metric(f, m->speedup);          fputc(',', f);
    write_metric(f, m->efficiency);       fputc('\n', f);
}

void csv_close(FILE *f) {
    if (f) {
        fclose(f);
    }
}
