/* timer.c — wall-clock timing implementation.
 *
 * Implements timer_start(), timer_stop(), timer_elapsed(), timer_reset()
 * using gettimeofday() for wall-clock timing (microsecond resolution).
 */

#include "timer.h"

#include <stddef.h>  /* NULL */

/* Converte struct timeval em segundos (double). */
static double tv_to_seconds(const struct timeval *tv) {
    return (double)tv->tv_sec + (double)tv->tv_usec / 1e6;
}

void timer_start(Timer *t) {
    gettimeofday(&t->start, NULL);
    t->elapsed = 0.0;
}

double timer_stop(Timer *t) {
    gettimeofday(&t->end, NULL);
    t->elapsed = tv_to_seconds(&t->end) - tv_to_seconds(&t->start);
    return t->elapsed;
}

double timer_elapsed(Timer *t) {
    if (t->elapsed > 0) {
        return t->elapsed;
    }
    struct timeval now;
    gettimeofday(&now, NULL);
    return tv_to_seconds(&now) - tv_to_seconds(&t->start);
}

void timer_reset(Timer *t) {
    t->elapsed = 0.0;
    gettimeofday(&t->start, NULL);
}
