#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

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
    unsigned long long local_pixels = 0;  

    unsigned long long chunk_size = r / size;
    unsigned long long start = rank * chunk_size;
    unsigned long long end = (rank == size - 1) ? r : start + chunk_size;


	unsigned long long pixels = 0;
	for (unsigned long long x = start; x < end; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		local_pixels += y;
		local_pixels %= k;
	}

	unsigned long long total_pixels = 0;
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k);
    }

	MPI_Finalize();
}
