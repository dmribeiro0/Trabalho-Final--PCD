/* timer.h — wall-clock timing primitive.
 *
 * Public interface for the timer module: start/stop/elapsed helpers that
 * provide raw elapsed time. Derived performance metrics (FLOPS, throughput,
 * speedup) are built on top of this in the metrics module (see metrics.h).
 *
 * Usa gettimeofday() para medir o tempo de parede (wall-clock). A resolução
 * é de microssegundos, suficiente para a granularidade dos experimentos.
 */
#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

/* Timer — armazena o instante inicial e final em struct timeval */
typedef struct {
    struct timeval start;
    struct timeval end;
    double elapsed;  /* tempo decorrido em segundos */
} Timer;

/* Inicia o timer */
void timer_start(Timer *t);

/* Para o timer e calcula o tempo decorrido (em segundos) */
double timer_stop(Timer *t);

/* Retorna o tempo decorrido sem parar (read-only, em segundos) */
double timer_elapsed(Timer *t);

/* Zera o timer */
void timer_reset(Timer *t);

#endif /* TIMER_H */
