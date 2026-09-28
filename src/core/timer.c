/* timer.c — wall-clock timing implementation.
 *
 * Implements timer_start(), timer_stop(), timer_elapsed(), timer_reset()
 * using clock_gettime(CLOCK_MONOTONIC) for high-resolution timing.
 */

#include "timer.h"

void timer_start(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
    t->elapsed = 0.0;
}

double timer_stop(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    t->elapsed = (t->end.tv_sec - t->start.tv_sec) +
                 (t->end.tv_nsec - t->start.tv_nsec) / 1e9;
    return t->elapsed;
}

double timer_elapsed(Timer *t) {
    if (t->elapsed > 0) {
        return t->elapsed;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - t->start.tv_sec) +
           (now.tv_nsec - t->start.tv_nsec) / 1e9;
}

void timer_reset(Timer *t) {
    t->elapsed = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}
