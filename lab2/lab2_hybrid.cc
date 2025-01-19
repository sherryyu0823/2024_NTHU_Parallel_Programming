#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
	int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

	unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long total_pixels = 0; 


    unsigned long long chunk_size = r / size;
    unsigned long long start = rank * chunk_size;
    unsigned long long end = (rank == size - 1) ? r : start + chunk_size;
	unsigned long long answer = 0;

	int num_threads = omp_get_max_threads();
	
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
    	unsigned long long local_pixels = 0;  


		#pragma omp for schedule(dynamic, 1000)
		for (unsigned long long x = start; x < end; x++) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			local_pixels += y;
			local_pixels %= k;
		}

		#pragma omp critical
		{
			answer += local_pixels;
			answer %= k;

		}
	}

    MPI_Reduce(&answer, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k);
    }

	MPI_Finalize();
}
