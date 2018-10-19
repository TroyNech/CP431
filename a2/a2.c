/* 
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: October 31, 2018
* 
* Usage: a2 size
* Description: 
*   Generates 2 arrays of random numbers with size $size, sorts them, then merges them in parallel
*   Merges using algorithm detailed at https://mylearningspace.wlu.ca/d2l/le/content/285513/viewContent/1586397/View
*   Reports time taken to merge and outputs merged array (to stdout)
* Limitations: 
*   Assumes that the input is correct
*   Assumes that n=2^k, k is an integer
* Compile like: mpicc a2.c -o a2 -std=c99 -lm
*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER_PROC 0

int comp(const void *a, const void *b);
void fill_and_sort(int *arr, long size);
void find_j_set(int *a_arr, int *b_arr, long og_arr_size, int *j_arr, long r, long k, long num_elem_in_first_part);
int find_b_size(int *j_arr, long start, long end);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    long og_arr_size = atof(argv[1]);
    long k = log2(og_arr_size); //number of elements in each partion
    long r = ceil((double) og_arr_size/k); //number of partions
    //first partition may have less elements if doesn't divide evenly
    long num_elem_in_first_part = (og_arr_size % r) ? k : og_arr_size % r;
    long parts_per_proc = r/num_procs; //partitions per proc
    long extra_parts = r%num_procs; //remaining partitions

    //vars for MPI_Scatterv
    int *j_arr, *a_arr, *b_arr, *send_counts, *displs;

    send_counts = (int *) malloc(sizeof(int)*num_procs);
    displs = (int *) malloc(sizeof(int)*num_procs);
            
    //assign remaining partitions to master proc
    send_counts[0] = parts_per_proc + extra_parts;
    displs[0] = 0;

    for (int i = 1; i < num_procs; i++) {
        //+1, -1 b/c need previous j value to calculate size of received array
        send_counts[i] = parts_per_proc + 1;
        displs[i] = send_counts[i-1] + displs[i-1] - 1;
    }

    //master create and sort 2 arrays of random numbers with size og_arr_size
    //then find J set
    if (rank == MASTER_PROC) {
        srand(time(NULL));

        a_arr = (int *) malloc(sizeof(int)*og_arr_size);
        b_arr = (int *) malloc(sizeof(int)*og_arr_size);
        fill_and_sort(a_arr, og_arr_size);
        fill_and_sort(b_arr, og_arr_size);

        j_arr = (int *) malloc(sizeof(int)*r);
        find_j_set(a_arr, b_arr, og_arr_size, j_arr, r, k, num_elem_in_first_part);
    } else {
        //+1 b/c need previous j value to calculate size of received array
        j_arr = (int *) malloc(sizeof(int)*parts_per_proc+1);
    }

    //master distributes J set
    //use parts_per_proc + extra_parts + 1 for recvcount in case MASTER_PROC is receiving extra partitions
    // or if extra_parts = 0, +1 is for other procs to receive previous j element
    MPI_Scatterv(j_arr, send_counts, displs, MPI_INT, j_arr, parts_per_proc + extra_parts + 1, MPI_INT, MASTER_PROC,
     MPI_COMM_WORLD);

    //each proc finds size of its B partition. Need to declare outside to avoid compiler warning
    int b_size;
    if (rank == MASTER_PROC) {
        b_size = find_b_size(j_arr, 0, parts_per_proc + extra_parts);
    } else {
        //pass start=1 b/c 0 is not part of proc's assigned set 
        b_size = find_b_size(j_arr, 1, parts_per_proc + 1);
    }

    //allocate A and B arrs
    if (rank != MASTER_PROC) {
        a_arr = (int *) malloc(sizeof(int)*k*parts_per_proc);
        b_arr = (int *) malloc(sizeof(int)*b_size);
    }

    //master gets all B sizes, which is value of B to send to each proc
    MPI_Gather(&b_size, 1, MPI_INT, send_counts, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

    //displs[0] already set from the last Scatterv
    if (rank == MASTER_PROC) {
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i-1] + send_counts[i-1];
        }
    }

    //master distributes B array
    MPI_Scatterv(b_arr, send_counts, displs, MPI_INT, b_arr, b_size, MPI_INT, MASTER_PROC,
     MPI_COMM_WORLD);

    //calculate send_counts to be the number of A elements each proc receives
    if (rank == MASTER_PROC) {
        send_counts[0] = num_elem_in_first_part;
        //displs[0] already set
        
        for (int i = 1; i < num_procs; i++) {
            send_counts[i] = k;
            displs[i] = send_counts[i-1] + displs[i-1];
        }
    }

    //master distributes A array
    //use num_elem_in_first_part as recv_count b/c all procs receive <= num_elem_in_first_part
    MPI_Scatterv(a_arr, send_counts, displs, MPI_INT, a_arr, num_elem_in_first_part, MPI_INT, MASTER_PROC,
     MPI_COMM_WORLD);

    //wait for all processes to start before beginning for timing reasons
    //declare start_time for all procs to avoid compiler warning later
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    //Each proc merge into their version of C (be aware that not all A partitions may have matching B partions)
    //  Also case where all A < all B, such that no proc except master has B, and master must check for this special case after going through A-B merge step

    //MPI_GatherV procs Cs into master (will be ordered in recv array according to rank, so will be in order)

    if (rank == MASTER_PROC) {
        double end_time = MPI_Wtime();
        double elapsed_seconds = end_time - start_time;

        printf("Took %.2f seconds\n", elapsed_seconds);

        //Master output C
    }
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}

//assumes srand already called, if desired
void fill_and_sort(int *arr, long size) {
    for (long i = 0; i < size; i++) {
        arr[i] = rand();
    }

    qsort(arr, size, sizeof(int), comp);
}

 //used by qsort
 //will result in ascending order
int comp(const void *a, const void *b)
{
    int *i = (int *) a;
    int *j = (int *) b;
    return *j - *i;
}

void find_j_set(int *a_arr, int *b_arr, long og_arr_size, int *j_arr, long r, long k, long num_elem_in_first_part) {
    long low = 0;
    long high, mid;

    //j is index of current J element
    for (long j = 0; j < r; j++) {
        high = og_arr_size;

        //finds first val in b_arr > a_arr[k*(j+1)]
        //no need to reset low var b/c searching for val > last low
        while (low < high) {
            mid = (low + high) / 2;
            if (b_arr[mid] <= a_arr[num_elem_in_first_part + k*(j) - 1]) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        //will give -1 J value in case that first available B value is > A[k*(j+1)]
        // or if all available B vals are < A[k*(j+1)]
        j_arr[j] = (low == high) ? -1 : low - 1;
    }
}

int find_b_size(int *j_arr, long start, long end) {   
    //find last B partition
    while (end > start - 1 && j_arr[end] == -1) {
        end--;
    }
    
    //size is 0 if no B partitions, else equal to the number of elements between
    //use start var instead of 0 b/c start is 0 for master and 1 for others (element 0 not assigned to them)
    int b_size = (end == start - 1) ? 0 : j_arr[end] - j_arr[0] + 1;

    return b_size;
}