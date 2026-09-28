# Makefile — PCD Research Framework
# Experimento: Impacto da divisão de dados no desempenho de filtros convolucionais
#
#   make            build the 'convolve_seq' binary (default, sequential)
#   make seq        build sequential version
#   make omp        build OpenMP version
#   make pthread    build Pthreads version
#   make cuda       build CUDA version (if nvcc available)
#   make clean      remove build artifacts
#
# Headers live in include/, sources in src/. Parallel phases use src/parallel/ and
# src/core/ for shared logic.

CC      = gcc
CXX     = g++
CFLAGS  = -Wall -Wextra -std=c11 -O3 -Iinclude/core -Iinclude/parallel -pthread
CFLAGS_OMP = $(CFLAGS) -fopenmp
LDFLAGS = -lm
LDFLAGS_OMP = $(LDFLAGS) -fopenmp

SRC_DIR   = src
INCLUDE_DIR = include
BIN_DIR   = bin

# Fontes do núcleo (lógica pura, reutilizável)
CORE_SRC = $(SRC_DIR)/core/image.c \
           $(SRC_DIR)/core/kernel.c \
           $(SRC_DIR)/core/matrix.c \
           $(SRC_DIR)/core/convolution.c \
           $(SRC_DIR)/core/timer.c \
           $(SRC_DIR)/core/metrics.c

# Fontes paralelos
CUDA_SRC = $(SRC_DIR)/parallel/convolution_cuda.cu

# Binários (separados por diretório para evitar conflito de nomes)
BIN_SEQ   = $(BIN_DIR)/seq/convolve_seq
BIN_OMP   = $(BIN_DIR)/omp/convolve_omp
BIN_PTHREAD = $(BIN_DIR)/pthread/convolve_pthread
BIN_CUDA  = $(BIN_DIR)/cuda/convolve_cuda

.PHONY: all seq omp pthread cuda clean test all_parallel

all: seq

# Criar diretórios de saída
$(BIN_DIR)/%:
	mkdir -p $(BIN_DIR)/$*

# Binário sequencial
$(BIN_SEQ): $(CORE_SRC) $(SRC_DIR)/core/main.c | $(BIN_DIR)/seq
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Binário OpenMP
$(BIN_OMP): $(CORE_SRC) $(SRC_DIR)/core/main.c | $(BIN_DIR)/omp
	$(CC) $(CFLAGS_OMP) $^ -o $@ $(LDFLAGS_OMP)

# Binário Pthreads
$(BIN_PTHREAD): $(CORE_SRC) $(SRC_DIR)/core/main.c | $(BIN_DIR)/pthread
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -pthread

# Binário CUDA (se nvcc disponível)
$(BIN_CUDA): $(CUDA_SRC) | $(BIN_DIR)/cuda
	nvcc -O3 -Iinclude/core -Iinclude/parallel -c -o $@.o $<
	nvcc -O3 -o $@ $@.o $(LDFLAGS)

# Alvos diretos
seq: $(BIN_SEQ)
omp: $(BIN_OMP)
pthread: $(BIN_PTHREAD)
cuda: $(BIN_CUDA)

# Todos os paralelos
all_parallel: omp pthread cuda

test:
	@echo "Run 'make seq' first, then execute:"
	@echo "./$(BIN_SEQ) -i images/input.ppm -k laplacian -b 32 -s 3"

clean:
	rm -rf $(BIN_DIR)
