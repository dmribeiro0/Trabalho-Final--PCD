# Filtros convolucionais paralelos — projeto experimental

Este projeto compara a aplicação de filtros convolucionais em imagens usando execução sequencial, OpenMP, Pthreads e CUDA. A pesquisa pretende investigar como a divisão dos dados afeta o desempenho e a eficiência energética. A hipótese é que blocos com boa localidade de memória tenham melhor desempenho que divisões cíclicas ou blocos muito pequenos.

## Estado atual

O programa aplica cinco filtros: laplaciano (`laplacian`), Sobel horizontal (`sobel_h`), Sobel vertical (`sobel_v`), nitidez (`sharpen`) e desfoque (`blur`). Ele mede o tempo de uma execução (agora com `gettimeofday`), estima operações por segundo (FLOPS/MFLOPS), calcula a aceleração (speedup) e a eficiência em relação a uma execução sequencial do mesmo filtro, e — quando disponível — mede a energia consumida via Intel RAPL para derivar MFLOPS/Watt.

Para experimentos repetidos existe um sistema de *cycle test*: executar um algoritmo N vezes sobre uma imagem e um kernel escolhidos, recolhendo as métricas de cada execução e gravando as N linhas num ficheiro CSV.

A divisão por blocos e as estratégias de distribuição ainda não estão implementadas. A opção `-b` é aceita e registrada nas métricas, mas não altera o processamento. O código dos filtros usa matrizes fixas de 3 × 3; use `-s 3`. O arquivo `config/defaults.yaml` registra parâmetros planejados, mas não é lido pelo programa.

## Métricas

O módulo de métricas (`src/core/metrics.c`) calcula:

- **FLOPS / MFLOPS** — número de operações de ponto flutuante por segundo. A contagem usa o frame completo: `total_flops = 2 × largura × altura × canais × k × k` (cada pixel de saída faz `k×k` multiplicações e `k×k` somas por canal; canais = 3, `k` = tamanho do kernel).
- **Speedup(P) = T_s / T_p(P)** — `T_s` é o tempo sequencial de baseline (média de um ciclo sequencial sobre a mesma imagem+kernel) e `T_p` o tempo da versão paralela.
- **Eficiência = Speedup(P) / P** — `P` é o número de threads usadas pela execução paralela, configurável via `-p` (runner) / `--threads` (executável único) / `THREADS=` (Makefile), com padrão 4. Para o modo sequencial `P = 1`. Não é definida para CUDA (onde `P` não corresponde a threads de CPU) e é registrada como `N/A`.
- **MFLOPS/Watt** — eficiência energética. A energia é medida via Intel RAPL (`/sys/class/powercap/intel-rapl:*/energy_uj`), lida imediatamente antes e depois da região cronometrada; `potência média = energia / tempo` e `MFLOPS/Watt = MFLOPS / potência`.

### Requisitos do RAPL

A leitura de energia requer uma CPU Intel com o subsistema *powercap* exposto em `/sys/class/powercap/intel-rapl`. Os contadores `energy_uj` costumam exigir privilégios de leitura (root ou capacidade adequada); em contentores ou máquinas sem RAPL a medição degrada graciosamente e os campos de energia são gravados como `N/A` (ou `null` no JSON). O módulo trata o *wraparound* do contador usando `max_energy_range_uj`.

## Cycle tests

Um *cycle test* executa um algoritmo N vezes e grava uma linha de métricas por execução num CSV.

```sh
make runner                       # Compila o runner de CPU (seq/omp/pthread)
make cuda-runner                  # Compila o runner CUDA (requer nvcc + GPU)

# Um ciclo (MODE, KERNEL, IMAGE, N, THREADS) -> CSV
make cycle-test MODE=omp KERNEL=sharpen IMAGE=images/a.jpg N=30 THREADS=8

# Um ciclo para cada combinação de modos de CPU x kernels
make cycle-all N=30 THREADS=8
make cycle-all CYCLE_MODES='seq omp pthread cuda' N=30   # inclui CUDA
```

Também é possível chamar o runner diretamente:

```sh
./bin/runner/runner -m pthread -i images/a.jpg -k blur -n 30 -p 8 -c results/csv/pthread_blur.csv
./bin/runner/runner_cuda -m cuda -i images/a.jpg -k blur -n 30   # variante CUDA
```

Opções do runner: `-m` modo, `-i` imagem, `-k` kernel, `-n` repetições, `-c` caminho do CSV (padrão `results/csv/<modo>_<kernel>.csv`), `-s` tamanho do kernel, `-p` número de threads (omp/pthread; padrão 4).

### Formato do CSV

Cada ciclo grava um cabeçalho seguido de N linhas (uma por execução) com as colunas:

```
timestamp,mode,image,image_w,image_h,kernel,kernel_size,run_index,threads_p,
time_s,flops,mflops,energy_joules,avg_power_w,mflops_per_watt,speedup,efficiency
```

Campos indisponíveis (ex.: energia sem RAPL, eficiência de CUDA) são gravados como `NA`. Se o ficheiro já existir, novas linhas são acrescentadas sem repetir o cabeçalho.

## Compilação e testes

São necessários GCC e `make`. Para CUDA, também são necessários `nvcc` e uma GPU NVIDIA disponível.

```sh
make                              # Compila a versão sequencial
make omp                          # Compila a versão OpenMP
make pthread                      # Compila a versão Pthreads
make cuda                         # Compila a versão CUDA
make test-all                     # Executa os quatro modos com os cinco filtros
make test-all MODES='seq omp pthread'  # Executa apenas os modos de CPU
make clean                        # Remove os executáveis de bin/
```

`make test-all` usa `images/a.jpg` e grava cada imagem e seu `metrics.json` em `results/<modo>/<filtro>/`. O comando para no primeiro erro. A versão CUDA exige GPU para executar; sem ela, use a variante de CPU indicada acima. As saídas em `results/` não são removidas por `make clean`.

## Uso

Cada modo tem seu executável em `bin/<modo>/convolve_<modo>`. Por exemplo:

```sh
./bin/seq/convolve_seq -i images/a.jpg -k laplacian -m seq -o saida_seq.png -t metricas_seq.json
./bin/omp/convolve_omp -i images/a.jpg -k blur -m omp -o saida_omp.png
./bin/pthread/convolve_pthread -i images/a.jpg -k sharpen -m pthread -o saida_pthread.png
```

As opções `-i` (imagem) e `-k` (filtro) são obrigatórias. Use `-m` para selecionar `seq`, `omp`, `pthread` ou `cuda`; `-h` mostra a ajuda. `-o` define o PNG e `-t` define o JSON de métricas. Nomes simples, como os dos exemplos, são gravados em `results/`. Para usar subdiretórios fora de `make test-all`, crie os diretórios intermediários antes da execução. Sem `-t`, as métricas ainda são gravadas em `results/metrics.json`. Sem `-o`, o caminho atual do programa resulta em `results/results/resultado.png`.

## Organização

- `src/core/` e `include/core/`: interface de linha de comando, carregamento de imagens, filtros, cronômetro (`timer.c`, agora com `gettimeofday`), convolução sequencial/OpenMP/Pthreads, e os módulos de métricas (`metrics.c`), energia RAPL (`energy.c`) e escrita de CSV (`csv.c`). `matrix.c` continua reservado para desenvolvimento futuro.
- `src/tools/runner.c`: driver dos *cycle tests* (compilado em `bin/runner/runner` e, com CUDA, `bin/runner/runner_cuda`).
- `src/parallel/convolution_cuda.cu`: implementação CUDA. Os arquivos OpenMP e Pthreads neste diretório não são usados pela compilação atual; essas implementações estão em `src/core/convolution.c`.
- `images/a.jpg`: imagem de exemplo usada por `make test-all`.
- `config/defaults.yaml`: proposta de configuração dos experimentos, ainda sem integração com a aplicação.
- `bin/`: executáveis gerados; `results/`: imagens e métricas geradas (JSON por execução única; CSVs de ciclo em `results/csv/`).

## Bibliografia

Títulos traduzidos livremente:

- TOUSIMOJARAD, Ashkan et al. Convolução de imagens 2D com três modelos de programação paralela no Xeon Phi.
- COSTA, Laura Caetano et al. Avaliação de desempenho de *k-means* com CPU e GPU, usando oneAPI e OpenMP, na detecção de intrusões em redes.
- DOS ANJOS, Paulo N. M. et al. Avaliando eficiência energética em padrões de algoritmos para computação científica e de alto desempenho.
