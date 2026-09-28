# Framework Experimental para Pesquisa em Convolutional Filters Paralelos

Este projeto é um framework experimental para pesquisa sobre o impacto da divisão
de dados no desempenho e eficiência energética de filtros convolucionais
paralelizados com **OpenMP**, **Pthreads**, e **CUDA**.

## Título do Projeto
**Impacto da divisão de dados no desempenho e na eficiência energética de filtros convolucionais paralelos**

## Aplicação
Aplicação paralela de filtros convolucionais (Laplaciano, Sobel, Sharpen, Blur).

## Pergunta de Pesquisa
Como a estratégia de divisão da imagem e o tamanho dos blocos afetam o desempenho e a eficiência energética de filtros convolucionais paralelizados?

## Hipótese
A divisão da imagem em blocos que preservem a localidade de memória apresentará melhor desempenho e eficiência energética do que distribuições cíclicas ou com blocos muito pequenos.

## Métricas
- Tempo de execução
- FLOPS (Floating Point Operations Per Second)
- Speedup (comparação com versão sequencial)

## Build

```sh
make            # build the sequential binary (default)
make seq        # build sequential version
make omp        # build OpenMP version
make pthread    # build Pthreads version
make cuda       # build CUDA version (requires nvcc)
make test-all   # run every mode/kernel combination; requires a CUDA GPU
make test-all MODES='seq omp pthread' # run CPU combinations without a GPU
make clean      # remove build artifacts
```

`make test-all` saves each image and its `metrics.json` in
`results/<mode>/<kernel>/`. It stops on the first failed run. CUDA needs
`nvcc` to build and an available NVIDIA GPU to run.

## Uso

```sh
./bin/seq/convolve_seq -i <imagem> -k <kernel> -m <modo> [-b block_size] [-s tamanho_kernel] [-o output] [-t metrics.json]
```

### Opções
- `-i, --image`: Caminho da imagem de entrada (obrigatório)
- `-k, --kernel`: Nome do kernel: laplacian, sobel_h, sobel_v, sharpen, blur (obrigatório)
- `-m, --mode`: Modo de execução: seq, omp, pthread, cuda (padrão: seq)
- `-b, --block`: Tamanho do bloco para divisão da imagem (padrão: 32)
- `-s, --size`: Tamanho do kernel (3, 5, 7, 9, ...) (padrão: 3)
- `-o, --output`: Caminho do arquivo de saída (padrão: resultado.png)
- `-t, --metrics`: Caminho do arquivo JSON para métricas (opcional)

### Exemplos

```sh
# Sequencial
./bin/seq/convolve_seq -i images/a.jpg -k laplacian -m seq -o result.png

# OpenMP
./bin/omp/convolve_omp -i images/a.jpg -k blur -m omp -b 64 -s 5 -o result.png

# Pthreads
./bin/pthread/convolve_pthread -i images/a.jpg -k sharpen -m pthread -o result.png
```

## Project Layout

```
├── Makefile                # Build system
├── README.md               # Este arquivo
├── config/                 # Arquivos de configuração de experimentos
│   └── defaults.yaml       # Valores padrão (block sizes, kernels)
├── src/
│   ├── core/               # Lógica pura, reutilizável
│   │   ├── convolution.c   # Implementação sequencial + stubs para paralelo
│   │   ├── image.c/.h      # Carregar/salvar imagens PGM/PPM
│   │   ├── kernel.c/.h     # Definição de kernels
│   │   ├── matrix.c/.h     # Operações de matriz genéricas
│   │   ├── timer.c/.h      # Timer de alta resolução
│   │   └── metrics.c/.h    # Métricas de desempenho
│   └── parallel/           # Implementações paralelas
│       ├── convolution_openmp.c
│       ├── convolution_pthread.c
│       └── convolution_cuda.cu
├── bin/                    # Binários gerados
│   ├── seq/                # Versão sequencial
│   ├── omp/                # Versão OpenMP
│   ├── pthread/            # Versão Pthreads
│   └── cuda/               # Versão CUDA
├── include/
│   ├── core/               # Headers da lógica pura
│   └── parallel/           # Headers do paralelismo
├── results/                # Saídas e métricas
└── images/                 # Imagens de exemplo (PPM/PGM)
```

## Módulos

| Arquivo               | Responsabilidade                                                    |
|----------------------|---------------------------------------------------------------------|
| `main.c`             | Entry point com CLI para experimentos automatizados.                |
| `image.h/.c`         | Container de imagem + PGM/PPM (Netpbm) load/save.                   |
| `kernel.h/.c`        | Tipo de kernel, kernels embutidos (Laplaciano, Sobel, Sharpen, Blur).|
| `matrix.h/.c`        | Tipo genérico de matriz + operações (add, multiply, transpose).     |
| `convolution.c`      | Motor de convolução (sequential + stubs para paralelo).             |
| `timer.h/.c`         | Timer wall-clock com alta resolução (clock_gettime).                |
| `metrics.h/.c`       | Métricas derivadas (FLOPS, throughput, speedup).                    |

Cada módulo tem um header em `include/core/` e uma implementação em `src/core/`.

## Bibliografia
- 2D Image Convolution using Three Parallel Programming Models on the Xeon Phi - TOUSIMOJARAD, Ashkan et al.
- Performance Evaluation of k-means using CPU and GPU with oneAPI and OpenMP for Network Intrusion Detection - COSTA, Laura Caetano et al.
- Avaliando eficiência energética em padrões de algoritmos para computação científica e de alto desempenho - DOS ANJOS, Paulo N. M. et al.
