/* image.h — image container and PGM/PPM (Netpbm) load/save interface.
 *
 * Public interface for the image module: the image data type and the
 * functions to read/write PGM (P5, grayscale) and PPM (P6, RGB) files.
 *
 * The module owner defines the types and function signatures here.
 */
#ifndef IMAGE_H
#define IMAGE_H

typedef struct {
    int width;
    int height;
    float *data;
} Image;

Image load_image(const char *filename);
void save_image(const char *filename, Image *img);
void free_image(Image *img);

#endif /* IMAGE_H */
