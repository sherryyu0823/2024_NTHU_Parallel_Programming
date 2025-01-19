//  g++  -O3 -lm -o hw1 hw1.cc
// srun -N1 -n2 ./wrapper.sh ./hw1  536869888 37.in 37.out
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <nvtx3/nvToolsExt.h>

// #define unsigned long long ll
#define min(x, y) (x<y ? x : y)
#define max(x, y) (x>y ? x : y)


// void odd_even_sort(float* local_data, int local_size, int rank, int size, int n, MPI_Comm comm) {

    // delete[] rdata;
    // delete[] ldata;
    // delete[] tmp;
// }

int main(int argc, char** argv){
    nvtxRangePush("main");
                                                                
    
    int n = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_File in_file, out_file;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);

    int local_size = n / size + (rank < n % size);
	int local_idx = n / size * rank + min(n % size, rank);

    float* local_data = new float[local_size];
    MPI_File_read_at(in_file, sizeof(float) * local_idx, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&in_file);
    
    nvtxRangePush("Computation");

    // first sorting for local pair
	boost::sort::spreadsort::float_sort(local_data, local_data + local_size);
    int phase = 0;
    float neighbor;
    int rsize = local_size - (rank + 1 == n % size);
    int lsize = local_size + (rank == (n % size));

    float* rdata = new float[rsize];
    float* ldata = new float[lsize];
    float* tmp = new float[local_size];

    phase = rank%2==1 ? 1:0;

    int t = size+1;


    while(t--)
    {

		// even phase
        if (phase == 0 && rank + 1 < size && rsize > 0 && local_size > 0) {
            // int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            //     int dest, int sendtag,
            //     void *recvbuf, int recvcount, MPI_Datatype recvtype,
            //     int source, int recvtag,
            //     MPI_Comm comm, MPI_Status *status)
            // communicate with right rank
            // MPI_Sendrecv(local_data + local_size - 1, 1, MPI_FLOAT, rank + 1, 0,
            //                 &neighbor, 1, MPI_FLOAT, rank + 1, 0, comm, MPI_STATUS_IGNORE);
            //  swapping(merge sort) if not sorted
            // printf("%f%f\n", local_data[local_size - 1] , neighbor);
            // if (local_data[local_size - 1] > neighbor) {
            nvtxRangePush("comm1");

            MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank + 1, 0,
                        rdata, rsize, MPI_FLOAT, rank + 1, 0, comm, MPI_STATUS_IGNORE);
            nvtxRangePop();
            
            int i = 0, r = 0, l = 0;
            while(i < local_size && r < rsize && l < local_size)
            {
                if(local_data[l] < rdata[r]){
                    tmp[i] = local_data[l]; i++; l++;
                    if (l == local_size) break;
                }
                else{
                    tmp[i] = rdata[r]; i++; r++;
                    if (r == rsize) break;
                }
            }
            if(l==local_size)
            {
                while (i != local_size){
                    tmp[i] = rdata[r]; i++; r++;}
            }
            if(r==rsize)
            {
                while (i != local_size){
                    tmp[i] = local_data[l]; i++; l++;}
            }
            std::swap(tmp, local_data);
            // }
            // check odd rank and swap with left item
        } 
        // odd phase
        else if (phase == 1 && rank != 0 && lsize > 0 && local_size > 0) {
        //    MPI_Sendrecv(local_data, 1, MPI_FLOAT, rank - 1, 0,
        //                     &neighbor, 1, MPI_FLOAT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            // if (local_data[0] < neighbor){
            nvtxRangePush("comm2");

            MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank - 1, 0,
                        ldata, lsize, MPI_FLOAT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            nvtxRangePop();

            int i = local_size-1, ll = lsize-1, l = local_size-1;
            while(i != -1 && ll != -1 && l != -1)
            {
                if(local_data[l] > ldata[ll]){
                    tmp[i] = local_data[l]; i--; l--;
                    // if (l == -1) break;
                }
                else{
                    tmp[i] = ldata[ll]; i--; ll--;
                    // if (ll == -1) break;
                }
            }
            if(l == -1)
            {
                while (i > -1){
                    tmp[i] = ldata[ll]; i--; ll--;}
            }

            if(ll == -1)
            {
                while (i > -1){
                    tmp[i] = local_data[l]; i--; l--;}
            }
            std::swap(tmp, local_data);
            // }
            
        }
        // MPI_Barrier(comm);
        phase = phase==1? 0:1;
    }
    nvtxRangePop();

	// odd_even_sort(local_data, local_size, rank, size, n, MPI_COMM_WORLD);
    nvtxRangePush("I/O");
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file);
	MPI_File_write_at(out_file, sizeof(float) * local_idx, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&out_file);
    nvtxRangePop();

	delete[] local_data;
	MPI_Finalize();
    nvtxRangePop();
	return 0;


}
