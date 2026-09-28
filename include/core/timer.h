/* timer.h — wall-clock timing primitive.
 *
 * Public interface for the timer module: start/stop/elapsed helpers that
 * provide raw elapsed time. Derived performance metrics (FLOPS, throughput,
 * speedup) are built on top of this in the metrics module (see metrics.h).
 *
 * Usando clock_gettime(CLOCK_MONOTONIC) para alta resolução e imunidade a
 * ajustes do sistema.
 */
#ifndef TIMER_H
#define TIMER_H

#include <time.h>

/* Timer结构 - armazena o tempo inicial e final em struct timespec */
typedef struct {
    struct timespec start;
    struct timespec end;
    double elapsed;  /* tempo decorrido em segundos */
} Timer;

/* Inicia o timer */
void timer_start(Timer *t);

/* Para o timer e calcula o tempo decorrido */
double timer_stop(Timer *t);

/* Retorna o tempo decorrido sem parar (read-only) */
double timer_elapsed(Timer *t);

/* Zera o timer */
void timer_reset(Timer *t);

#endif /* TIMER_H */
