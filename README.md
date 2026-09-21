# 2D Convolutional Filters — PCD Final Project

Sequential C implementation of 2D convolutional image filters, structured so the
compute engine can later be parallelized with **OpenMP**, **Pthreads**, and
**CUDA**.

This repository currently holds the **project structure** only. Each module's
types, interfaces, and implementation are left to the module owner.

## Build

```sh
make        # build the 'convolve' binary
make test   # build and run the test suite
make clean  # remove build artifacts
```

## Project Layout

```
├── Makefile
├── README.md
├── include/            public headers (one per module)
│   ├── image.h
│   ├── kernel.h
│   ├── matrix.h
│   ├── convolution.h
│   ├── timer.h
│   └── metrics.h
├── src/                implementations
│   ├── main.c
│   ├── image.c
│   ├── kernel.c
│   ├── matrix.c
│   ├── convolution.c
│   ├── timer.c
│   └── metrics.c
├── tests/
│   └── test_main.c     test runner
└── images/             sample PGM/PPM inputs
```

## Modules — what each file is for

| File               | Responsibility                                                    |
|--------------------|-------------------------------------------------------------------|
| `main.c`           | Program entry point; wires the pipeline (args, load, convolve, save). |
| `image.h/.c`       | Image container + PGM/PPM (Netpbm) load/save.                     |
| `kernel.h/.c`      | Kernel type, built-in kernels, and user kernel loading.           |
| `matrix.h/.c`      | Generic matrix type + operations (add, multiply, transpose, outer product); used by kernel. |
| `convolution.h/.c` | Convolution compute engine (general + separable); to be parallelized later. |
| `timer.h/.c`       | Wall-clock timing primitive (raw elapsed time).                   |
| `metrics.h/.c`     | Derived performance metrics (FLOPS, throughput, speedup) from times + op counts. |
| `tests/test_main.c`| Test runner.                                                      |

Each module has a header in `include/` and an implementation in `src/`. Pick a
module and define its interface and implementation.
