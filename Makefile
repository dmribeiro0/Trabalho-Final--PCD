# Makefile — PCD Research Framework
# Experimento: Impacto da divisão de dados no desempenho de filtros convolucionais
#
#   make            build the 'convolve_seq' binary (default, sequential)
#   make main       build and execute o main.c (sequential, OpenMP e pthread)
#   make seq        build sequential version
#   make omp        build OpenMP version
#   make pthread    build Pthreads version
#   make cuda       build CUDA version (if nvcc available)
#   make runner     build the CPU cycle-test runner (seq/omp/pthread)
#   make cuda-runner build the CUDA cycle-test runner (needs nvcc + GPU)
#   make cycle-test MODE=.. KERNEL=.. IMAGE=.. N=.. THREADS=..  run one cycle -> CSV
#   make cycle-all  N=.. THREADS=..    run cycles for all CPU modes x kernels -> CSVs
#   make test-all   run all convolution tests (all modes x all kernels)
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
RESULTS_DIR = results

# Fontes do núcleo (lógica pura, reutilizável)
CORE_SRC = $(SRC_DIR)/core/image.c \
           $(SRC_DIR)/core/kernel.c \
           $(SRC_DIR)/core/matrix.c \
           $(SRC_DIR)/core/convolution.c \
           $(SRC_DIR)/core/timer.c \
           $(SRC_DIR)/core/metrics.c \
           $(SRC_DIR)/core/energy.c \
           $(SRC_DIR)/core/csv.c

# Fonte do runner (driver de cycle tests)
RUNNER_SRC = $(SRC_DIR)/tools/runner.c

# Fontes paralelos
CUDA_SRC = $(SRC_DIR)/parallel/convolution_cuda.cu
CUDA_CORE_OBJ = $(patsubst $(SRC_DIR)/core/%.c,$(BIN_DIR)/cuda/%.o,$(CORE_SRC) $(SRC_DIR)/core/main.c)

# Binários (separados por diretório para evitar conflito de nomes)
BIN_SEQ   = $(BIN_DIR)/seq/convolve_seq
BIN_OMP   = $(BIN_DIR)/omp/convolve_omp
BIN_PTHREAD = $(BIN_DIR)/pthread/convolve_pthread
BIN_CUDA  = $(BIN_DIR)/cuda/convolve_cuda
BIN_RUNNER = $(BIN_DIR)/runner/runner
BIN_RUNNER_CUDA = $(BIN_DIR)/runner/runner_cuda

# Kernels
KERNELS = laplacian sobel_h sobel_v sharpen blur
# Modos
MODES = seq omp pthread cuda

.PHONY: all main seq omp pthread cuda clean test-all all_parallel runner cuda-runner cycle-test cycle-all

all: seq

# Compila e executa o programa principal, que percorre os três modos de CPU.
main: omp
	./$(BIN_OMP)

# Binário sequencial
$(BIN_SEQ): $(CORE_SRC) $(SRC_DIR)/core/main.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Binário OpenMP
$(BIN_OMP): $(CORE_SRC) $(SRC_DIR)/core/main.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS_OMP) $^ -o $@ $(LDFLAGS_OMP)

# Binário Pthreads
$(BIN_PTHREAD): $(CORE_SRC) $(SRC_DIR)/core/main.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -pthread

# Runner de cycle tests (CPU: seq + omp + pthread num único binário).
# Compilado com OpenMP + pthread para que todos os modos de CPU funcionem.
$(BIN_RUNNER): $(CORE_SRC) $(RUNNER_SRC)
	mkdir -p $(@D)
	$(CC) $(CFLAGS_OMP) $^ -o $@ $(LDFLAGS_OMP) -pthread

# Binário CUDA (se nvcc disponível)
$(BIN_DIR)/cuda/%.o: $(SRC_DIR)/core/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -DHAVE_CUDA -c $< -o $@

$(BIN_CUDA): $(CUDA_SRC) $(CUDA_CORE_OBJ)
	mkdir -p $(@D)
	nvcc -O3 -Iinclude/core -c -o $@.o $<
	nvcc -O3 -o $@ $@.o $(CUDA_CORE_OBJ) $(LDFLAGS) -Xcompiler -pthread

# Runner com CUDA (opcional; requer nvcc + GPU NVIDIA). Reutiliza os objetos
# do núcleo + runner.o compilados com -DHAVE_CUDA e -fopenmp, linkados via nvcc
# com o .cu. Só é construído por `make cuda-runner`; o build normal não depende
# de nvcc.
RUNNER_CUDA_OBJ = $(patsubst $(SRC_DIR)/core/%.c,$(BIN_DIR)/runner/%.o,$(CORE_SRC)) \
                  $(BIN_DIR)/runner/runner.o

$(BIN_DIR)/runner/%.o: $(SRC_DIR)/core/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS_OMP) -DHAVE_CUDA -c $< -o $@

$(BIN_DIR)/runner/runner.o: $(RUNNER_SRC)
	mkdir -p $(@D)
	$(CC) $(CFLAGS_OMP) -DHAVE_CUDA -c $< -o $@

$(BIN_RUNNER_CUDA): $(CUDA_SRC) $(RUNNER_CUDA_OBJ)
	mkdir -p $(@D)
	nvcc -O3 -Iinclude/core -c -o $(BIN_DIR)/runner/convolution_cuda.o $<
	nvcc -O3 -o $@ $(BIN_DIR)/runner/convolution_cuda.o $(RUNNER_CUDA_OBJ) $(LDFLAGS) -Xcompiler -pthread -Xcompiler -fopenmp

# Alvos diretos
seq: $(BIN_SEQ)
omp: $(BIN_OMP)
pthread: $(BIN_PTHREAD)
cuda: $(BIN_CUDA)
runner: $(BIN_RUNNER)
cuda-runner: $(BIN_RUNNER_CUDA)

# Todos os paralelos
all_parallel: seq omp pthread cuda

# --- Cycle tests (executar um algoritmo N vezes; métricas -> CSV) ---
# Parâmetros (com valores padrão): MODE, KERNEL, IMAGE, N, THREADS, CSV
MODE    ?= seq
KERNEL  ?= blur
IMAGE   ?= images/a.jpg
N       ?= 10
THREADS ?= 4
CSV     ?= $(RESULTS_DIR)/csv/$(MODE)_$(KERNEL).csv

# Um ciclo de N execuções para (MODE, KERNEL, IMAGE). Usa o runner CUDA quando
# MODE=cuda (requer `make cuda-runner`), senão o runner de CPU. THREADS define
# o número de threads (P) para omp/pthread.
cycle-test:
	@if [ "$(MODE)" = "cuda" ]; then \
		$(MAKE) cuda-runner; \
		./$(BIN_RUNNER_CUDA) -m $(MODE) -i $(IMAGE) -k $(KERNEL) -n $(N) -p $(THREADS) -c "$(CSV)"; \
	else \
		$(MAKE) runner; \
		./$(BIN_RUNNER) -m $(MODE) -i $(IMAGE) -k $(KERNEL) -n $(N) -p $(THREADS) -c "$(CSV)"; \
	fi

# Um ciclo para cada combinação de modos de CPU x kernels (um CSV por combinação).
# Para incluir CUDA, faça `make cycle-all CYCLE_MODES='seq omp pthread cuda'`
# após `make cuda-runner`. THREADS define o número de threads (P) para omp/pthread.
CYCLE_MODES ?= seq omp pthread
cycle-all: runner
	@set -eu; for mode in $(CYCLE_MODES); do \
		for kernel in $(KERNELS); do \
			echo "== Cycle: mode=$$mode kernel=$$kernel N=$(N) threads=$(THREADS) =="; \
			if [ "$$mode" = "cuda" ]; then \
				./$(BIN_RUNNER_CUDA) -m $$mode -i $(IMAGE) -k $$kernel -n $(N) -p $(THREADS) \
					-c "$(RESULTS_DIR)/csv/$${mode}_$${kernel}.csv"; \
			else \
				./$(BIN_RUNNER) -m $$mode -i $(IMAGE) -k $$kernel -n $(N) -p $(THREADS) \
					-c "$(RESULTS_DIR)/csv/$${mode}_$${kernel}.csv"; \
			fi; \
		done; \
	done
	@echo "Cycle tests concluídos. CSVs em $(RESULTS_DIR)/csv/"

# Pipeline completo de testes: todas as combinações (modos x kernels)
test-all: $(MODES)
	@echo "Running all convolution tests..."
	@set -eu; for mode in $(MODES); do \
		for kernel in $(KERNELS); do \
			out_dir="$(RESULTS_DIR)/$${mode}/$${kernel}"; \
			out_file="$${mode}/$${kernel}/output.png"; \
			metrics_file="$${mode}/$${kernel}/metrics.json"; \
			echo "Testing: mode=$$mode, kernel=$$kernel"; \
			mkdir -p "$$out_dir"; \
			./bin/$${mode}/convolve_$${mode} -i images/a.jpg -k $$kernel -m $$mode -o "$$out_file" -t "$$metrics_file"; \
			test -s "$$out_dir/output.png"; \
			test -s "$$out_dir/metrics.json"; \
		done; \
	done
	@echo "All tests completed. Results in $(RESULTS_DIR)/"

clean:
	rm -rf $(BIN_DIR)
