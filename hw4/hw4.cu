#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o);

float _max(float a, float b) { return a > b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

__global__ void QKDotAndScalarKernel(float *out, float *q, float *k, int N, int d, float scalar) {
    __shared__ float q_shared[32 * 32]; 
    __shared__ float k_shared[32 * 32];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int row = blockIdx.x * blockDim.x + tx;
    int col = blockIdx.y * blockDim.y + ty;
    if (row >= N || col >= N) return;
    float sum = 0.0f;

    for (int tile = 0; tile < (d + blockDim.x - 1) / blockDim.x; tile++) {
        int shared_idx = tx * blockDim.y + ty;

        // 將 q 和 k 的對應部分載入到 shared memory
        if (row < N && tile * blockDim.x + ty < d) {
            q_shared[shared_idx] = q[row * d + tile * blockDim.x + ty];
        } else {
            q_shared[shared_idx] = 0.0f; 
        }

        if (col < N && tile * blockDim.x + tx < d) {
            k_shared[shared_idx] = k[col * d + tile * blockDim.x + tx];
        } else {
            k_shared[shared_idx] = 0.0f; 
        }

        __syncthreads();
        // 計算 Q 和 K 的內積
        for (int t = 0; t < blockDim.x; t++) {
            int q_idx = tx * blockDim.x + t;  
            int k_idx = t * blockDim.x + ty; 
            sum += q_shared[q_idx] * k_shared[k_idx];
        }

        __syncthreads(); // 確保所有 thread 完成當前 tile 的計算
    }

    if (row < N && col < N) {
        out[row * N + col] = sum * scalar;
    }
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < N && col < N) {
    //     float value = 0.0f;
    //     for (int t = 0; t < d; t++) {
    //         value += q[row * d + t] * k[col * d + t];
    //     }
    //     out[row * N + col] = value * scalar;
    // }
}

__global__ void RowMaxKernel(float *out, float *in, int N, int stride) {
    // int row = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < N) {
    //     float max_val = -FLT_MAX;
    //     for (int col = 0; col < stride; col++) {
    //         max_val = fmaxf(max_val, in[row * stride + col]);
    //     }
    //     out[row] = max_val;
    // }
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < stride; col++) {
            if (col < stride) { // 確保在有效範圍內
                max_val = fmaxf(max_val, in[row * stride + col]);
            }
        }
        out[row] = max_val;
    }
}

__global__ void MinusMaxAndExpKernel(float *out, float *in, float *row_max, int N, int stride) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float val;
    
    if (row < N && col < stride) {

        val = in[row * stride + col] - row_max[row];
        val = fmaxf(val, -20.0f);
        val = fminf(val, 20.0f);
        out[row * stride + col] = expf(val);
    }
}


__global__ void RowSumKernel(float *out, float *in, int N, int stride) {
    // int row = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < N) {
    //     float sum = 0.0f;
    //     for (int col = 0; col < stride; col++) {
    //         sum += in[row * stride + col];
    //     }
    //     out[row] = sum;
    // }
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < stride; col++) {
            if (col < stride) { 
                sum += in[row * stride + col];
            }
        }
        out[row] = sum;
    }
}

__global__ void UpdateOutputKernel(float *out, float *q, float *k, float *v, float *row_sum, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < d) {
        float result = 0.0f;
        float safe_row_sum = fmaxf(row_sum[row], 1e-8f);
        for (int t = 0; t < N; t++) {
            result += q[row * N + t] * v[t * d + col] / safe_row_sum;
        }
        out[row * d + col] = result;
    }
}

void flash_attention(float *q, float *k, float *v, float *o) {
    float *d_q, *d_k, *d_v, *d_o;
    float *d_sij, *d_row_max, *d_pij, *d_row_sum;
    int size_matrix = N * N * sizeof(float);
    int size_vector = N * sizeof(float);
    int size_output = N * d * sizeof(float);

    cudaMalloc(&d_q, size_output);
    cudaMalloc(&d_k, size_output);
    cudaMalloc(&d_v, size_output);
    cudaMalloc(&d_o, size_output);

    cudaMalloc(&d_sij, size_matrix);
    cudaMalloc(&d_row_max, size_vector);
    cudaMalloc(&d_pij, size_matrix);
    cudaMalloc(&d_row_sum, size_vector);

    cudaMemcpy(d_q, q, size_output, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size_output, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_output, cudaMemcpyHostToDevice);
    
    // 使用 cudaOccupancyMaxPotentialBlockSize 獲取最佳block大小
    int blockSizeQKDot, blockSizeRowMax, blockSizeMinusMax, blockSizeRowSum, blockSizeUpdateOutput;
    int minGridSizeQKDot, minGridSizeRowMax, minGridSizeMinusMax, minGridSizeRowSum, minGridSizeUpdateOutput;

    cudaOccupancyMaxPotentialBlockSize(&minGridSizeQKDot, &blockSizeQKDot, QKDotAndScalarKernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeRowMax, &blockSizeRowMax, RowMaxKernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeMinusMax, &blockSizeMinusMax, MinusMaxAndExpKernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeRowSum, &blockSizeRowSum, RowSumKernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeUpdateOutput, &blockSizeUpdateOutput, UpdateOutputKernel, 0, 0);

    // 確保block大小不超過32x32
    blockSizeQKDot = min(blockSizeQKDot, 32);
    blockSizeRowMax = min(blockSizeRowMax, 32);
    blockSizeMinusMax = min(blockSizeMinusMax, 32);
    blockSizeRowSum = min(blockSizeRowSum, 32);
    blockSizeUpdateOutput = min(blockSizeUpdateOutput, 32);
    // printf("%d\n", blockSizeRowMax);
    // dim3 block_qk_dot(16, 16);
    // dim3 grid_qk_dot((N + block_qk_dot.x - 1) / block_qk_dot.x, (N + block_qk_dot.y - 1) / block_qk_dot.y);

    // dim3 block_1d_row_max(blockSizeRowMax, 1);
    // dim3 grid_1d_row_max((N + block_1d_row_max.x - 1) / block_1d_row_max.x, 1);

    // dim3 block_minus_max(blockSizeMinusMax, blockSizeMinusMax);
    // dim3 grid_minus_max((N + block_minus_max.x - 1) / block_minus_max.x, 
    //                     (N + block_minus_max.y - 1) / block_minus_max.y);

    // dim3 block_1d_row_sum(blockSizeRowSum, 1);
    // dim3 grid_1d_row_sum((N + block_1d_row_sum.x - 1) / block_1d_row_sum.x, 1);

    // dim3 block_update_output(blockSizeUpdateOutput, blockSizeUpdateOutput);
    // dim3 grid_update_output((N + block_update_output.x - 1) / block_update_output.x, 
    //                          (d + block_update_output.y - 1) / block_update_output.y);
    dim3 block_qk_dot(8, 8);
    dim3 grid_qk_dot(N/8, N/8);

    dim3 block_1d_row_max(128, 1);
    dim3 grid_1d_row_max(N/128, 1);

    dim3 block_minus_max(32, 32);
    dim3 grid_minus_max(N/32, N/32);

    dim3 block_1d_row_sum(128, 1);
    dim3 grid_1d_row_sum(N/128, 1);

    dim3 block_update_output(16, 16);
    dim3 grid_update_output(N/16, N/16);
    
    QKDotAndScalarKernel<<<grid_qk_dot, block_qk_dot>>>(d_sij, d_q, d_k, N, d, 1.0f / sqrtf(d));
    // cudaDeviceSynchronize();

    RowMaxKernel<<<grid_1d_row_max, block_1d_row_max>>>(d_row_max, d_sij, N, N);
    // cudaDeviceSynchronize();

    MinusMaxAndExpKernel<<<grid_minus_max, block_minus_max>>>(d_pij, d_sij, d_row_max, N, N);
    // cudaDeviceSynchronize();

    RowSumKernel<<<grid_1d_row_sum, block_1d_row_sum>>>(d_row_sum, d_pij, N, N);
    // cudaDeviceSynchronize();

    UpdateOutputKernel<<<grid_update_output, block_update_output>>>(d_o, d_pij, d_k, d_v, d_row_sum, N, d);

    cudaMemcpy(o, d_o, size_output, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_sij);
    cudaFree(d_row_max);
    cudaFree(d_pij);
    cudaFree(d_row_sum);
}

// 以下代碼保持不變
void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    // free(Q);
    // free(K);
    // free(V);
    // free(O);

    fclose(file);
}


int main(int argc, char *argv[]) {
    nvtxRangePush("Main");

    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    for (int i = 0; i < B; i++) {
        flash_attention(
            Q + (i * N * d),
            K + (i * N * d),
            V + (i * N * d),
            O + (i * N * d)
        );
    }
    nvtxRangePop();

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}
