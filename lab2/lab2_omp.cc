#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

	int num_threads = omp_get_max_threads();  
	unsigned long long* partial_sum = new unsigned long long[num_threads]; // 为每个线程分配局部和

	for (int i = 0; i < num_threads; i++) {
		partial_sum[i] = 0;
	}

	#pragma omp parallel
	{
		int tid = omp_get_thread_num(); 

		#pragma omp for schedule(static, 1000)
		for (unsigned long long x = 0; x < r; x++) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			partial_sum[tid] += y;
			partial_sum[tid] %= k; 
		}
	}

	unsigned long long answer = 0;
	for (int i = 0; i < num_threads; i++) {
		answer += partial_sum[i];
	}

	printf("%llu\n", (4 * answer) % k);

	delete[] partial_sum;

	return 0;
}
