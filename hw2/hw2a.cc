// g++ -O3 -lm  -pthread -mavx512f hw2a.cc -o hw2a -lpng
// srun -n1 -c4 nsys profile -o hw2a_p4_fast04 ./hw2a fast04.png 1813 -2 0 -2 0 1347 1651
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h> // AVX-512 header for _mm512d
#include <nvtx3/nvToolsExt.h>

double y_offset, x_offset, left, right, lower, upper;
int width, height, iters;
int t_height = 0; // 用於記錄當前行的索引
int *image;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void *mandelbrot(void *arg)
{
    nvtxRangePush("Thread Mandelbrot Computation");
    while (t_height < height)
    {
        int row_local;
        if (t_height >= height)
        {
            pthread_mutex_unlock(&mutex);
            break;
        }
        row_local = t_height++;
        pthread_mutex_unlock(&mutex);

        double y0 = row_local * y_offset + lower;

        int n = 8;
        int idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int iter = n - 1;
        __m512d x0 = _mm512_set_pd(
            (7 < width) ? 7 * x_offset + left : 0,
            (6 < width) ? 6 * x_offset + left : 0,
            (5 < width) ? 5 * x_offset + left : 0,
            (4 < width) ? 4 * x_offset + left : 0,
            (3 < width) ? 3 * x_offset + left : 0,
            (2 < width) ? 2 * x_offset + left : 0,
            (1 < width) ? 1 * x_offset + left : 0,
            (0 < width) ? 0 * x_offset + left : 0);
        __m512d y0_vec = _mm512_set1_pd(y0);
        __m512d x = _mm512_setzero_pd();
        __m512d y = _mm512_setzero_pd();
        __m512d x2 = _mm512_mul_pd(x, x);
        __m512d y2 = _mm512_mul_pd(y, y);
        __m512d length_squared = _mm512_setzero_pd();

        int repeats[n];
        memset(repeats, 0, n * sizeof(int));

        while (idx[0] < width || idx[1] < width || idx[2] < width || idx[3] < width ||
               idx[4] < width || idx[5] < width || idx[6] < width || idx[7] < width)
        {
            __m512d temp = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0);
            y = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_mul_pd(x, y)), y0_vec);
            x = temp;
            x2 = _mm512_mul_pd(x, x);
            y2 = _mm512_mul_pd(y, y);
            length_squared = _mm512_add_pd(x2, y2);

            for (int i = 0; i < n; i++)
            {
                if (idx[i] < width)
                {
                    repeats[i]++;
                    if (repeats[i] >= iters || length_squared[i] >= 4.0)
                    {
                        image[row_local * width + idx[i]] = repeats[i];
                        idx[i] = ++iter; // process the next pixel
                        if (idx[i] < width)
                        {
                            x0[i] = idx[i] * x_offset + left;
                            x[i] = 0;
                            y[i] = 0;
                            x2[i] = 0;
                            y2[i] = 0;
                            length_squared[i] = 0;
                            repeats[i] = 0;
                        }
                    }
                }
            }
        }
    }
    nvtxRangePop();
    return NULL;
}

int main(int argc, char **argv)
{
    nvtxRangePush("main");

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    pthread_t threads[ncpus];

    assert(argc == 9);

    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    y_offset = (upper - lower) / height;
    x_offset = (right - left) / width;

    image = (int *)malloc(width * height * sizeof(int));
    assert(image);

    nvtxRangePush("Mandelbrot");
    for (int i = 0; i < ncpus; i++)
    {
        pthread_create(&threads[i], NULL, mandelbrot, NULL);
    }
    nvtxRangePop();
    for (int i = 0; i < ncpus; i++)
    {
        pthread_join(threads[i], NULL);
    }

    nvtxRangePush("I/O Operation - Write PNG");
    write_png(filename, iters, width, height, image);
    nvtxRangePop();

    free(image);
    pthread_mutex_destroy(&mutex);
    nvtxRangePop();
    return 0;
}
