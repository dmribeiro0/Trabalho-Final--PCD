/* kernel.h — convolution kernel interface.
 *
 * Public interface for the kernel module: the kernel data type, the built-in
 * kernels (e.g. box, gaussian, sobel, sharpen), and loading user-defined
 * kernels from a file. Separability is a design concern for this module.
 *
 * The module owner defines the types and function signatures here.
 */
#ifndef KERNEL_H
#define KERNEL_H

#define KERNEL_SIZE 3
#define KERNEL_AREA (KERNEL_SIZE * KERNEL_SIZE)

typedef struct {
    int size;
    float data[KERNEL_AREA];
} Kernel;

void get_convolution_filters(Kernel filters[5]);

#endif /* KERNEL_H */
