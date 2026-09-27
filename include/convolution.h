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

typedef struct {
    int width;
    int height;
    float *data;
} Image;

// 3X3
typedef struct {
    int size;
    float data[9]; 
} Kernel;

void apply_convolution(Image *input, Kernel *kernel, Image *output);
void get_convolution_filters(Kernel filters[5]);

#endif /* CONVOLUTION_H */
