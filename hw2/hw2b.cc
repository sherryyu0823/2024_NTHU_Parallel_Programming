// g++ -fopenmp -O3 -mavx512f -o hw2b hw2b.cc -lpng -I/usr/include/mpi -L/usr/lib -lmpi
// srun -n2 -c2 ./wrapper.sh ./hw2b fast06.png 5455 -1.25 0 0 1.25 483 631
// mpicc -fopenmp -O3 -mavx512f -o hw2b hw2b.cc -lpng -lstdc++

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
#include <mpi.h>
#include <omp.h>
#include <immintrin.h> // AVX-512 header for _mm512d
#include <nvtx3/nvToolsExt.h>
#include <algorithm>

double y_offset, x_offset, left, right, lower, upper;
int width, height, iters;
int *image;

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

void mandelbrot(int start_row, int end_row)
{

    #pragma omp parallel for schedule(dynamic)
    for (int row_local = start_row; row_local < end_row; row_local++)
    {
        double y0 = row_local * y_offset + lower;

        int n = 8;  
        int idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int iter = n - 1;

        // for (int i = 0; i < width; i += 8)
        // {
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

        int repeats[8] = {0};
    nvtxRangePush("Thread Mandelbrot Computation");

        while (idx[0] < width || idx[1] < width || idx[2] < width || idx[3] < width ||
                idx[4] < width || idx[5] < width || idx[6] < width || idx[7] < width)
        {
            // temp = x * x - y * y + x0;
            // y = 2 * x * y + y0;
            __m512d temp = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0);
            // y = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_mul_pd(x, y)), y0_vec);
            y = _mm512_fmadd_pd(_mm512_mul_pd(x, y), _mm512_set1_pd(2.0), y0_vec);
            x = temp;
            x2 = _mm512_mul_pd(x, x);
            y2 = _mm512_mul_pd(y, y);
            length_squared = _mm512_add_pd(x2, y2);

            for (int k = 0; k < n; k++)
            {
                if (idx[k] < width)
                {
                    repeats[k]++;
                    if (repeats[k] >= iters || length_squared[k] >= 4.0)
                    {
                        image[row_local * width + idx[k]] = repeats[k];
                        idx[k] = ++iter;  // process the next pixel
                        if (idx[k] < width)
                        {
                            x0[k] = idx[k] * x_offset + left;
                            x[k] = 0 ;y[k] = 0; x2[k] = 0; y2[k] = 0;
                            length_squared[k] = 0;
                            repeats[k] = 0;
                        }
                    }
                }
            }
        }
        // }
    }

    nvtxRangePop();
}

int main(int argc, char **argv)
{
    nvtxRangePush("main");

    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // int rows_per_proc = height / size;
    // int start_row = rank * rows_per_proc;
    // int end_row = (rank == size - 1) ? height : start_row + rows_per_proc;
    int rows_per_proc = height / size;
    int extra_rows = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);

    nvtxRangePush("Mandelbrot Computation");
    mandelbrot(start_row, end_row);
    nvtxRangePop();

    int *full_image = NULL;
    if (rank == 0)
    {
        full_image = (int *)malloc(width * height * sizeof(int));
        assert(full_image);
    }

    // int *recvcounts = NULL;
    // int *displs = NULL;
    // if (rank == 0)
    // {
    //     recvcounts = (int *)malloc(size * sizeof(int));
    //     displs = (int *)malloc(size * sizeof(int));
    //     for (int i = 0; i < size; ++i)
    //     {
    //         int start = i * rows_per_proc;
    //         int end = (i == size - 1) ? height : start + rows_per_proc;
    //         recvcounts[i] = (end - start) * width;
    //         displs[i] = start * width;
    //     }
    // }
    int *recvcounts = NULL;
int *displs = NULL;
if (rank == 0)
{
    recvcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; ++i)
    {
        int start = i * rows_per_proc + std::min(i, extra_rows);
        int end = start + rows_per_proc + (i < extra_rows ? 1 : 0);
        
        recvcounts[i] = (end - start) * width; 
        displs[i] = offset * width;            
        offset += (end - start);               
    }
}


    // Gather all the results
    MPI_Gatherv(image + start_row * width, (end_row - start_row) * width, MPI_INT,
                full_image, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // 主进程写PNG
    if (rank == 0)
    {
        write_png(filename, iters, width, height, full_image);
        free(full_image);
        free(recvcounts);
        free(displs);
    }

    free(image);
    MPI_Finalize();

    nvtxRangePop();
    return 0;
}