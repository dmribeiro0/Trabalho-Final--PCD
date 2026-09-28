/* image.c — image module implementation.
 *
 * Implements the interface declared in include/image.h: loading and saving
 * PGM (P5) and PPM (P6) Netpbm images.
 *
 * TODO: implement this module.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image.h"
#include <stdlib.h>

Image load_image(const char *filename) {
    Image img = {0, 0, NULL};
    int channels;
    unsigned char *raw = stbi_load(filename, &img.width, &img.height, &channels, 3);
    
    if (raw) {
        img.data = (float *)malloc(img.width * img.height * 3 *sizeof(float));
        for (int i = 0; i < img.width * img.height * 3; i++) {
            img.data[i] = (float)raw[i];
        }
        stbi_image_free(raw);
    }
    return img;
}

void save_image(const char *filename, Image *img) {
    unsigned char *raw = (unsigned char *)malloc(img->width * img->height * 3);
    
    for (int i = 0; i < img->width * img->height * 3; i++) {
        float val = img->data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        raw[i] = (unsigned char)val;
    }
    
    stbi_write_png(filename, img->width, img->height, 3, raw, img->width * 3);
    free(raw);
}

void free_image(Image *img) {
    if (img->data) {
        free(img->data);
        img->data = NULL;
    }
}