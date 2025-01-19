#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>

unsigned long long r;
unsigned long long k;
unsigned long long ncpus;
unsigned long long *partial_sum;

void* calc(void* thread_id) {
    int tid = *(int*)thread_id;

    unsigned long long chunk_size = r / ncpus;
    unsigned long long start = tid * chunk_size;
    unsigned long long end = (tid == ncpus - 1) ? r : start + chunk_size;
    unsigned long long y;

    for (unsigned long long x = start; x < end; x++) {
        if (r * r >= x * x) {
            y = ceil(sqrtl(r * r - x * x));
            partial_sum[tid] += y;
            partial_sum[tid] %= k;
        }
    }
    pthread_exit(nullptr);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    r = atoll(argv[1]);
    k = atoll(argv[2]);

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ncpus = CPU_COUNT(&cpuset);

    partial_sum = new unsigned long long[ncpus];
    for (unsigned long long i = 0; i < ncpus; i++) {
        partial_sum[i] = 0; 
    }

    int* ids = new int[ncpus];
    pthread_t threads[ncpus];

    for (unsigned long long i = 0; i < ncpus; i++) {
        ids[i] = i;
        int ret = pthread_create(&threads[i], nullptr, calc, (void*)&ids[i]);
        if (ret != 0) {
            fprintf(stderr, "Error creating thread %llu\n", i);
            return 1;
        }
    }
    for (unsigned long long i = 0; i < ncpus; i++) {
        int ret = pthread_join(threads[i], nullptr);
        if (ret != 0) {
            fprintf(stderr, "Error joining thread %llu\n", i);
            return 1;
        }
    }

    unsigned long long answer = 0;
    for (unsigned long long i = 0; i < ncpus; i++) {
        answer += partial_sum[i];
    }

    printf("%llu\n", (4 * answer) % k);

    delete[] partial_sum;
    delete[] ids;

    return 0;
}
