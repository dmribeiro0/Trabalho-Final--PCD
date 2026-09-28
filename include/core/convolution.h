/* convolution.h — convolution compute engine interface.
 *
 * Public interface for the convolution module: the general and separable
 * convolution routines. This is the compute-heavy core intended to be
 * parallelized later (OpenMP / Pthreads / CUDA).
 *
 * The module owner defines the function signatures here.
 */
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "image.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// Funções sequenciais e paralelas (implementadas em src/core e src/parallel)
void apply_convolution_sequential(Image *input, Kernel *kernel, Image *output);
void apply_convolution_openmp(Image *input, Kernel *kernel, Image *output);
void apply_convolution_pthread(Image *input, Kernel *kernel, Image *output);
void apply_convolution_cuda(Image *input, Kernel *kernel, Image *output);

#ifdef __cplusplus
}
#endif

#endif /* CONVOLUTION_H */
