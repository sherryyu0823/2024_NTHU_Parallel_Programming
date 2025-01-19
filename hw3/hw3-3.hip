#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <omp.h>
//======================
#define DEV_NO 0
#define B 64
#define HB 32
const int INF = ((1 << 30) - 1);

int n, m, orig_n;
int* Dist;
hipDeviceProp_t prop;

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

__global__ void Phase3(int *d_Dist, int id, int pitch, int offset){
    // skipping the blocks from phase 1 & 2
	if (blockIdx.x == id || (blockIdx.y + offset) == id) return;

    int x = threadIdx.x;
    int y = threadIdx.y;

    // index for d_Dist
    int global_x = x + blockIdx.x * B;
    int global_y = y + (blockIdx.y + offset) * B;

    __shared__ int Shared_D[B][B];

    // load data from golbal memory to shared memory
    // execute 32*32 data per block(32*32 threads in it)
    Shared_D[y][x] = d_Dist[global_y*pitch+ global_x];
    Shared_D[y + HB][x] = d_Dist[(global_y + HB) * pitch+ global_x];
    Shared_D[y][x + HB] = d_Dist[global_y * pitch+ global_x + HB];
    Shared_D[y + HB][x + HB] = d_Dist[(global_y + HB) * pitch+ global_x + HB];

	int i = y + id * B;
	int j = x + blockIdx.x * B;

	__shared__ int Col3D[B][B];
	Col3D[y][x] = d_Dist[i * pitch+ j];
	Col3D[y + HB][x] = d_Dist[(i + HB) * pitch+ j];
	Col3D[y][x + HB] = d_Dist[i * pitch+ (j + HB)];
	Col3D[y + HB][x + HB] = d_Dist[(i + HB) * pitch+ (j + HB)];

	// load the target block of same row 
	i = y + (blockIdx.y + offset) * B;
	j = x + id * B;

	__shared__ int RowD[B][B];
	RowD[y][x] = d_Dist[i * pitch+ j];
	RowD[y + HB][x] = d_Dist[(i + HB) * pitch+ j];
	RowD[y][x + HB] = d_Dist[i * pitch+ (j + HB)];
	RowD[y + HB][x + HB] = d_Dist[(i + HB) * pitch+ (j + HB)];
	__syncthreads();

	#pragma unroll 48
	for (int k = 0; k < B; ++k) {
		Shared_D[y][x] = min(Shared_D[y][x], RowD[y][k] + Col3D[k][x]);
		Shared_D[y + HB][x] = min(Shared_D[y + HB][x], RowD[y + HB][k] + Col3D[k][x]);
		Shared_D[y][x + HB] = min(Shared_D[y][x + HB], RowD[y][k] + Col3D[k][x + HB]);
		Shared_D[y + HB][x + HB] = min(Shared_D[y + HB][x + HB], RowD[y + HB][k] + Col3D[k][x + HB]);
	}

    // shared memory --> global memory
    d_Dist[global_y * pitch+ global_x] = Shared_D[y][x];
    d_Dist[(global_y + HB) * pitch+ global_x] = Shared_D[y + HB][x];
    d_Dist[global_y * pitch+ global_x + HB] = Shared_D[y][x + HB];
    d_Dist[(global_y + HB) * pitch+ global_x + HB] = Shared_D[y + HB][x + HB];
}



int main(int argc, char* argv[]) {
    input(argv[1]);
    
    int *d_Dist[2];
    // cudaMalloc((void **) &d_Dist, n * n * sizeof(int));
    // cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    int BlockNum = n/B;
    hipHostRegister(Dist, n * n * sizeof(int), hipHostMallocDefault);

    
    //  Total amount of shared memory per block: 49152 bytes
    dim3 grid1(1, 1);
    dim3 grid2(BlockNum, 1);
    dim3 NumofThreads(32, 32);

    
    #pragma omp parallel num_threads(2)
    {
        int threadID = omp_get_thread_num();
		hipSetDevice(threadID);
        dim3 grid3(BlockNum, BlockNum/2);

        int offset;
        if(threadID == 0) offset = 0; else offset = BlockNum/2;
        if (threadID == 1 && (BlockNum % 2 == 1)) grid3.y++;

        hipMalloc((void **) &d_Dist[threadID], n * n * sizeof(int));
        // 只傳輸需要的，盡量減少傳輸量
        hipMemcpy(d_Dist[threadID]+ offset * B * n, Dist + offset * B * n, 
                    grid3.y * B * n * sizeof(int), hipMemcpyHostToDevice);
        

        // cudaMallocPitch((void**)&d_Dist[threadId], &pitch, n * sizeof(int), n);
        // cudaMemcpy2D(d_Dist[threadId], pitch, Dist, n * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
        
        for(int id = 0; id < BlockNum; id++){
            
            if (id >= offset && id < offset + grid3.y) {
				hipMemcpy(Dist + id * B * n, d_Dist[threadID] + id * B * n, B * n * sizeof(int), hipMemcpyDeviceToHost);

            }
            #pragma omp barrier
			if (id < offset || id >= offset + grid3.y) {
				hipMemcpy(d_Dist[threadID] + id * B * n, Dist + id * B * n, B * n * sizeof(int), hipMemcpyHostToDevice);

            }
            
            Phase1 <<<grid1, NumofThreads>>> (d_Dist[threadID], id, n);
            // #pragma omp barrier
            Phase2 <<<grid2, NumofThreads>>> (d_Dist[threadID], id, n);
            // #pragma omp barrier
            Phase3 <<<grid3, NumofThreads>>> (d_Dist[threadID], id, n, offset);

            // #pragma omp barrier

            // if((id+1) >= offset && (id+1) < offset + grid3.y) {
            //     if(threadID == 0) {
            //         cudaMemcpyPeer(d_Dist[1] + (id + 1) * B * n, 1, 
            //             d_Dist[0] + (id + 1) * B * n, 0, B * n * sizeof(int));
            //     }
            //     else {
            //         cudaMemcpyPeer(d_Dist[0] + (id + 1) * B * n, 0, d_Dist[1] + (id + 1) * B * n, 1, B * n * sizeof(int));
            //     }
            // }
            // #pragma omp barrier
   
        }
        // cudaGetDeviceProperties(&prop, DEV_NO);
        // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

        // block_FW(B);
        hipMemcpy(Dist + offset * B * n, d_Dist[threadID]+ offset * B * n,
                    grid3.y * B * n * sizeof(int), hipMemcpyDeviceToHost);
        // cudaMemcpy2D(Dist, n * sizeof(int), d_Dist, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
        // cudaFree(d_Dist);
        output(argv[2]);

    }
    
    return 0;
}
