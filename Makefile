# Makefile — 2D Convolutional Filters (PCD Final Project)
#
#   make            build the 'convolve' binary (default)
#   make test       build and run the test suite
#   make clean      remove build artifacts
#
# Headers live in include/, sources in src/. Later parallel phases
# (OpenMP, Pthreads, CUDA) add their own targets/flags here.

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2 -Iinclude
LDFLAGS = -lm

SRC_DIR  = src
TEST_DIR = tests
BIN      = convolve
TEST_BIN = run_tests

# Library sources (everything except main.c), reused by the binary and tests.
LIB_SRCS = $(SRC_DIR)/image.c \
           $(SRC_DIR)/kernel.c \
           $(SRC_DIR)/matrix.c \
           $(SRC_DIR)/convolution.c \
           $(SRC_DIR)/timer.c \
           $(SRC_DIR)/metrics.c

MAIN_SRC = $(SRC_DIR)/main.c
TEST_SRC = $(TEST_DIR)/test_main.c

.PHONY: all build test clean
all: build
build: $(BIN)

$(BIN): $(MAIN_SRC) $(LIB_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(LIB_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(BIN) $(TEST_BIN)
