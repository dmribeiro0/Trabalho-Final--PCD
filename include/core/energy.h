/* energy.h — energy measurement via Intel RAPL (Linux powercap sysfs).
 *
 * Lê os contadores de energia em /sys/class/powercap/intel-rapl:* (domínios
 * de pacote e, quando presente, DRAM). Uma amostra é tirada antes e depois da
 * região cronometrada; energy_delta_joules() devolve a energia consumida no
 * intervalo, tratando o wraparound do contador.
 *
 * Degradação graciosa: se o sysfs não existir ou não for legível (CPU não
 * Intel, falta de permissão, contêiner sem acesso), energy_available()
 * devolve false e as leituras marcam-se como indisponíveis.
 */
#ifndef ENERGY_H
#define ENERGY_H

#include <stdbool.h>
#include <stdint.h>

/* Máximo de domínios RAPL de topo que monitorizamos. */
#define ENERGY_MAX_DOMAINS 16

typedef struct {
    bool     available;                        /* leitura válida? */
    int      n_domains;                        /* domínios lidos */
    uint64_t energy_uj[ENERGY_MAX_DOMAINS];    /* leitura por domínio (µJ) */
    uint64_t max_range_uj[ENERGY_MAX_DOMAINS]; /* wraparound por domínio (µJ) */
} EnergySample;

/* Verdadeiro se pelo menos um domínio RAPL é legível. */
bool energy_available(void);

/* Tira uma amostra instantânea de todos os domínios RAPL de topo. */
EnergySample energy_read(void);

/* Energia consumida (Joules) entre duas amostras, somando todos os domínios
 * e tratando o wraparound do contador. Devolve valor negativo (sentinela)
 * se qualquer amostra for inválida. */
double energy_delta_joules(const EnergySample *begin, const EnergySample *end);

#endif /* ENERGY_H */
