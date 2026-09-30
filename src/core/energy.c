/* energy.c — Intel RAPL energy measurement (Linux powercap sysfs).
 *
 * Enumera os domínios de topo /sys/class/powercap/intel-rapl:<i> e lê os seus
 * contadores energy_uj (µJ). O sentinela de indisponibilidade é ENERGY_NA_J.
 */
#include "energy.h"

#include <stdio.h>
#include <stdint.h>

#define RAPL_BASE "/sys/class/powercap"
#define ENERGY_NA_J (-1.0)

/* Lê um inteiro sem sinal de um ficheiro sysfs. Devolve false em falha. */
static bool read_u64_file(const char *path, uint64_t *out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return false;
    }
    unsigned long long v = 0;
    int n = fscanf(f, "%llu", &v);
    fclose(f);
    if (n != 1) {
        return false;
    }
    *out = (uint64_t)v;
    return true;
}

EnergySample energy_read(void) {
    EnergySample s;
    s.available = false;
    s.n_domains = 0;

    for (int i = 0; i < ENERGY_MAX_DOMAINS; i++) {
        char energy_path[256];
        char range_path[256];
        snprintf(energy_path, sizeof(energy_path),
                 RAPL_BASE "/intel-rapl:%d/energy_uj", i);
        snprintf(range_path, sizeof(range_path),
                 RAPL_BASE "/intel-rapl:%d/max_energy_range_uj", i);

        uint64_t energy = 0;
        if (!read_u64_file(energy_path, &energy)) {
            /* Domínios de topo são contíguos a partir de 0; o primeiro
             * ausente encerra a enumeração. */
            break;
        }

        uint64_t range = 0;
        if (!read_u64_file(range_path, &range)) {
            range = 0; /* sem info de wraparound para este domínio */
        }

        int idx = s.n_domains;
        s.energy_uj[idx] = energy;
        s.max_range_uj[idx] = range;
        s.n_domains++;
    }

    s.available = (s.n_domains > 0);
    return s;
}

bool energy_available(void) {
    EnergySample s = energy_read();
    return s.available;
}

double energy_delta_joules(const EnergySample *begin, const EnergySample *end) {
    if (!begin || !end || !begin->available || !end->available) {
        return ENERGY_NA_J;
    }
    /* Os domínios devem coincidir entre as duas amostras. */
    if (begin->n_domains != end->n_domains || begin->n_domains == 0) {
        return ENERGY_NA_J;
    }

    double total_uj = 0.0;
    for (int i = 0; i < begin->n_domains; i++) {
        uint64_t b = begin->energy_uj[i];
        uint64_t e = end->energy_uj[i];
        uint64_t delta;

        if (e >= b) {
            delta = e - b;
        } else {
            /* Wraparound: o contador reiniciou. */
            uint64_t range = end->max_range_uj[i];
            if (range > 0) {
                delta = (range - b) + e;
            } else {
                /* Sem info de range; ignora este domínio neste intervalo. */
                continue;
            }
        }
        total_uj += (double)delta;
    }

    return total_uj / 1e6; /* µJ -> J */
}
