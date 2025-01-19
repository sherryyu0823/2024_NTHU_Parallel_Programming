#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <nvtx3/nvToolsExt.h>

//======================
#define DEV_NO 0
#define B 64
#define HB 32
const int INF = ((1 << 30) - 1);

int n, m, orig_n;
int* Dist;
// cudaDeviceProp prop;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // padding
    orig_n = n;
    if(orig_n % B != 0) n = orig_n + (B - orig_n % B);

    Dist = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) Dist[i*n+j] = 0;
            else Dist[i*n+j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         if (Dist[i][j] >= INF) Dist[i][j] = INF;
    //     }
    //     fwrite(Dist[i], sizeof(int), n, outfile);
    // }
    
    // only write the original n*n 
    for (int i = 0; i < orig_n; ++i) {
		fwrite(Dist + i * n, sizeof(int), orig_n, outfile);
	}
    fclose(outfile);
}


__global__ void Phase1(int *d_Dist, int id, size_t pitch){
    
    int x = threadIdx.x;
    int y = threadIdx.y;

    // index for d_Dist
    int global_x = x + id * B;
    int global_y = y + id * B;

    __shared__ int shared_D[B][B];

    // load data from golbal memory to shared memory
    // execute 32*32 data per block(32*32 threads in it)
    shared_D[y][x] = d_Dist[global_y*pitch+ global_x];
    shared_D[y + HB][x] = d_Dist[(global_y + HB) * pitch+ global_x];
    shared_D[y][x + HB] = d_Dist[global_y * pitch+ global_x + HB];
    shared_D[y + HB][x + HB] = d_Dist[(global_y + HB) * pitch+ global_x + HB];
    __syncthreads();


    #pragma unroll 48
    for(int i = 0; i < B; i++) {
        shared_D[y][x] = min(shared_D[y][x], shared_D[y][i] + shared_D[i][x]);
        shared_D[y + HB][x] = min(shared_D[y + HB][x], shared_D[y + HB][i] + shared_D[i][x]);
        shared_D[y][x + HB] = min(shared_D[y][x + HB], shared_D[y][i] + shared_D[i][x + HB]);
        shared_D[y + HB][x + HB] = min(shared_D[y + HB][x + HB], shared_D[y + HB][i] + shared_D[i][x + HB]);

        __syncthreads();
    }

    // shared memory --> global memory
    d_Dist[global_y * pitch+ global_x] = shared_D[y][x];
    d_Dist[(global_y + HB) * pitch+ global_x] = shared_D[y + HB][x];
    d_Dist[global_y * pitch+ global_x + HB] = shared_D[y][x + HB];
    d_Dist[(global_y + HB) * pitch+ global_x + HB] = shared_D[y + HB][x + HB];
}

__global__ void Phase2(int *d_Dist, int id, size_t pitch){

    // skipping the pivot block (from phase 1)
	if (blockIdx.x == id) return;

	int x = threadIdx.x;
    int y = threadIdx.y;

    // pivot block from phase1
    int global_x = x + id * B;
    int global_y = y + id * B;

	__shared__ int pivotD[B][B];
	pivotD[y][x] = d_Dist[global_y*pitch+ global_x];
    pivotD[y + HB][x] = d_Dist[(global_y + HB) * pitch+ global_x];
    pivotD[y][x + HB] = d_Dist[global_y * pitch+ global_x + HB];
    pivotD[y + HB][x + HB] = d_Dist[(global_y + HB) * pitch+ global_x + HB];

	// load the target block of same column into shared memory
	int i = y + id * B;
	int j = x + blockIdx.x * B;

	__shared__ int ColD[B][B];
	ColD[y][x] = d_Dist[i * pitch+ j];
	ColD[y + HB][x] = d_Dist[(i + HB) * pitch+ j];
	ColD[y][x + HB] = d_Dist[i * pitch+ (j + HB)];
	ColD[y + HB][x + HB] = d_Dist[(i + HB) * pitch+ (j + HB)];

	// load the target block of same row 
	i = y + blockIdx.x * B;
	j = x + id * B;

	__shared__ int RowD[B][B];
	RowD[y][x] = d_Dist[i * pitch+ j];
	RowD[y + HB][x] = d_Dist[(i + HB) * pitch+ j];
	RowD[y][x + HB] = d_Dist[i * pitch+ (j + HB)];
	RowD[y + HB][x + HB] = d_Dist[(i + HB) * pitch+ (j + HB)];

	__syncthreads();

	#pragma unroll 48
	for (int k = 0; k < B; ++k) {
		// using cuda min
		ColD[y][x] = min(ColD[y][x], pivotD[y][k] + ColD[k][x]);
		ColD[y + HB][x] = min(ColD[y + HB][x], pivotD[y + HB][k] + ColD[k][x]);
		ColD[y][x + HB] = min(ColD[y][x + HB], pivotD[y][k] + ColD[k][x + HB]);
		ColD[y + HB][x + HB] = min(ColD[y + HB][x + HB], pivotD[y + HB][k] + ColD[k][x + HB]);

		RowD[y][x] = min(RowD[y][x], RowD[y][k] + pivotD[k][x]);
		RowD[y + HB][x] = min(RowD[y + HB][x], RowD[y + HB][k] + pivotD[k][x]);
		RowD[y][x + HB] = min(RowD[y][x + HB], RowD[y][k] + pivotD[k][x + HB]);
		RowD[y + HB][x + HB] = min(RowD[y + HB][x + HB], RowD[y + HB][k] + pivotD[k][x + HB]);	
	}

    // shared memory --> global memory
	i = y + blockIdx.x * B;
	j = x + id * B;
	d_Dist[i * pitch+ j] = RowD[y][x];
	d_Dist[(i + HB) * pitch+ j] = RowD[y + HB][x];
	d_Dist[i * pitch+ (j + HB)] = RowD[y][x + HB];
	d_Dist[(i + HB) * pitch+ (j + HB)] = RowD[y + HB][x + HB];

	i = y + id * B;
	j = x + blockIdx.x * B;
	d_Dist[i * pitch+ j] = ColD[y][x];
	d_Dist[(i + HB) * pitch+ j] = ColD[y + HB][x];
	d_Dist[i * pitch+ (j + HB)] = ColD[y][x + HB];
	d_Dist[(i + HB) * pitch+ (j + HB)] = ColD[y + HB][x + HB];   
}

__global__ void Phase3(int *d_Dist, int id, size_t pitch){
    // skipping the blocks from phase 1 & 2
	if (blockIdx.x == id || blockIdx.y == id) return;

    int x = threadIdx.x;
    int y = threadIdx.y;

    // index for d_Dist
    int global_x = x + blockIdx.x * B;
    int global_y = y + blockIdx.y * B;

    __shared__ int shared_D[B][B];

    // load data from golbal memory to shared memory
    // execute 32*32 data per block(32*32 threads in it)
    shared_D[y][x] = d_Dist[global_y*pitch+ global_x];
    shared_D[y + HB][x] = d_Dist[(global_y + HB) * pitch+ global_x];
    shared_D[y][x + HB] = d_Dist[global_y * pitch+ global_x + HB];
    shared_D[y + HB][x + HB] = d_Dist[(global_y + HB) * pitch+ global_x + HB];

	int i = y + id * B;
	int j = x + blockIdx.x * B;

	__shared__ int ColD[B][B];
	ColD[y][x] = d_Dist[i * pitch+ j];
	ColD[y + HB][x] = d_Dist[(i + HB) * pitch+ j];
	ColD[y][x + HB] = d_Dist[i * pitch+ (j + HB)];
	ColD[y + HB][x + HB] = d_Dist[(i + HB) * pitch+ (j + HB)];

	// load the target block of same row 
	i = y + blockIdx.y * B;
	j = x + id * B;

	__shared__ int RowD[B][B];
	RowD[y][x] = d_Dist[i * pitch+ j];
	RowD[y + HB][x] = d_Dist[(i + HB) * pitch+ j];
	RowD[y][x + HB] = d_Dist[i * pitch+ (j + HB)];
	RowD[y + HB][x + HB] = d_Dist[(i + HB) * pitch+ (j + HB)];
	__syncthreads();

	#pragma unroll 48
	for (int k = 0; k < B; ++k) {
		shared_D[y][x] = min(shared_D[y][x], RowD[y][k] + ColD[k][x]);
		shared_D[y + HB][x] = min(shared_D[y + HB][x], RowD[y + HB][k] + ColD[k][x]);
		shared_D[y][x + HB] = min(shared_D[y][x + HB], RowD[y][k] + ColD[k][x + HB]);
		shared_D[y + HB][x + HB] = min(shared_D[y + HB][x + HB], RowD[y + HB][k] + ColD[k][x + HB]);
	}

    // shared memory --> global memory
    d_Dist[global_y * pitch+ global_x] = shared_D[y][x];
    d_Dist[(global_y + HB) * pitch+ global_x] = shared_D[y + HB][x];
    d_Dist[global_y * pitch+ global_x + HB] = shared_D[y][x + HB];
    d_Dist[(global_y + HB) * pitch+ global_x + HB] = shared_D[y + HB][x + HB];
}



int main(int argc, char* argv[]) {

    nvtxRangePush("Main");
    nvtxRangePush("I/O Time");
    input(argv[1]);
    nvtxRangePop();
    
    int *d_Dist;
    // cudaMalloc((void **) &d_Dist, n * n * sizeof(int));
    // cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    int BlockNum = n/B;
    cudaHostRegister(Dist, n * n * sizeof(int), cudaHostAllocDefault);

    size_t pitch;
	cudaMallocPitch((void**)&d_Dist, &pitch, n * sizeof(int), n);
	cudaMemcpy2D(d_Dist, pitch, Dist, n * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);

    //  Total amount of shared memory per block: 49152 bytes
    dim3 grid1(1, 1);
    dim3 grid2(BlockNum, 1);
    dim3 grid3(BlockNum, BlockNum);
    dim3 NumofThreads(32, 32);

    size_t pitchints = pitch/sizeof(int);

    nvtxRangePush("Cmp Time");

    for(int id = 0; id < BlockNum; id++){
        Phase1 <<<grid1, NumofThreads>>> (d_Dist, id, pitchints);
        Phase2 <<<grid2, NumofThreads>>> (d_Dist, id, pitchints);
        Phase3 <<<grid3, NumofThreads>>> (d_Dist, id, pitchints);
    }
    // cudaDeviceSynchronize();
    nvtxRangePop();

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

    // block_FW(B);
    // cudaMemcpy(Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy2D(Dist, n * sizeof(int), d_Dist, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);

    nvtxRangePush("I/O Time");
    output(argv[2]);
    nvtxRangePop();

    nvtxRangePop();

    return 0;
}

