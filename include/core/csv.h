/* csv.h — CSV writer for cycle-test metrics.
 *
 * Um "ciclo" executa um algoritmo N vezes; cada execução produz uma linha de
 * métricas (ver metrics.h). Este módulo abre um ficheiro CSV (criando os
 * diretórios pais e escrevendo o cabeçalho quando o ficheiro é novo/vazio),
 * acrescenta uma linha por execução e fecha o ficheiro.
 *
 * Colunas:
 *   timestamp,mode,image,image_w,image_h,kernel,kernel_size,run_index,
 *   threads_p,time_s,flops,mflops,energy_joules,avg_power_w,mflops_per_watt,
 *   speedup,efficiency
 *
 * Campos indisponíveis (METRIC_NA) são escritos como "NA".
 */
#ifndef CSV_H
#define CSV_H

#include <stdio.h>
#include "metrics.h"

/* Abre o CSV em 'path' para escrita/append. Cria os diretórios pais e escreve
 * o cabeçalho se o ficheiro ainda não existir ou estiver vazio. Devolve NULL
 * em falha. */
FILE *csv_open(const char *path);

/* Acrescenta uma linha ao CSV a partir de um registo de métricas. */
void csv_append_row(FILE *f, const Metrics *m);

/* Fecha o CSV. */
void csv_close(FILE *f);

#endif /* CSV_H */
