#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <pthread.h>

int V, E;
int ncpus;

int **D;
int* D1;
const int INF = (1 << 30) - 1;

pthread_barrier_t barrier;

void input(char *input_file){
    FILE *file = fopen(input_file, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    D1 = (int*)malloc(V * V * sizeof(int));
    D = (int**)malloc(V * sizeof(int*));

    for (int i = 0; i < V; i++) {
        D[i] = D1 + i * V;
    }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) D[i][j] = 0;   
            else D[i][j] = INF;      
        }
    }

    int pair[3];
    for (int i = 0; i < E; i++){
        fread(pair, sizeof(int), 3, file);
        D[pair[0]][pair[1]] = pair[2];
    }

    fclose(file);

    return;
}

void output(char *output_file){

    FILE *file = fopen(output_file, "w");

    for (int i = 0; i < V; i++){
        fwrite(D[i], sizeof(int), V, file);
    }
    fclose(file);
    return;
}

void *FW(void *args){

    int tid = *(int *)args;
    int start = tid * (V / ncpus) + std::min(tid, V % ncpus);
    int end = start + V / ncpus + (tid < V % ncpus);


    for (int k = 0; k < V; k++){
        for (int i = start; i < end; i ++) {
            for (int j = 0; j < V; j++) {
                D[i][j] = std::min(D[i][j], D[i][k] + D[k][j]);
            }
        }

        pthread_barrier_wait(&barrier);
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]){
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ncpus = CPU_COUNT(&cpuset);
    pthread_barrier_init(&barrier, NULL, ncpus);

    input(argv[1]);

    pthread_t* threads = (pthread_t*)malloc(ncpus * sizeof(pthread_t));
    int* thread_ids = (int*)malloc(ncpus * sizeof(int));
    for (int i = 0; i < ncpus; ++i) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, FW, (void*)&thread_ids[i]);
    }

    for (int i = 0; i < ncpus; ++i)
        pthread_join(threads[i], NULL);
    
    output(argv[2]);

    pthread_exit(NULL);
}