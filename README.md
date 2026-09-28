# Filtros convolucionais paralelos — projeto experimental

Este projeto compara a aplicação de filtros convolucionais em imagens usando execução sequencial, OpenMP, Pthreads e CUDA. A pesquisa pretende investigar como a divisão dos dados afeta o desempenho e a eficiência energética. A hipótese é que blocos com boa localidade de memória tenham melhor desempenho que divisões cíclicas ou blocos muito pequenos.

## Estado atual

O programa aplica cinco filtros: laplaciano (`laplacian`), Sobel horizontal (`sobel_h`), Sobel vertical (`sobel_v`), nitidez (`sharpen`) e desfoque (`blur`). Ele mede o tempo de uma execução, estima operações por segundo (FLOPS) e calcula a aceleração em relação a uma execução sequencial do mesmo filtro.

A divisão por blocos, as estratégias de distribuição, as repetições de experimentos e a medição de energia ainda não estão implementadas. A opção `-b` é aceita e registrada nas métricas, mas não altera o processamento. O código dos filtros usa matrizes fixas de 3 × 3; use `-s 3`. O arquivo `config/defaults.yaml` registra parâmetros planejados, mas não é lido pelo programa.

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

- `src/core/` e `include/core/`: interface de linha de comando, carregamento de imagens, filtros, cronômetro e convolução sequencial, OpenMP e Pthreads. `matrix.c` e `metrics.c` ainda são módulos reservados para desenvolvimento futuro; as métricas atuais são calculadas em `main.c`.
- `src/parallel/convolution_cuda.cu`: implementação CUDA. Os arquivos OpenMP e Pthreads neste diretório não são usados pela compilação atual; essas implementações estão em `src/core/convolution.c`.
- `images/a.jpg`: imagem de exemplo usada por `make test-all`.
- `config/defaults.yaml`: proposta de configuração dos experimentos, ainda sem integração com a aplicação.
- `bin/`: executáveis gerados; `results/`: imagens e métricas geradas.

## Bibliografia

Títulos traduzidos livremente:

- TOUSIMOJARAD, Ashkan et al. Convolução de imagens 2D com três modelos de programação paralela no Xeon Phi.
- COSTA, Laura Caetano et al. Avaliação de desempenho de *k-means* com CPU e GPU, usando oneAPI e OpenMP, na detecção de intrusões em redes.
- DOS ANJOS, Paulo N. M. et al. Avaliando eficiência energética em padrões de algoritmos para computação científica e de alto desempenho.
