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
void merge_arrays(int *a_arr, int *b_arr, int *c_arr, int *j_arr, long num_proc_parts, long extra_parts, long k,
             long num_elem_in_first_part);
void append_array(int *source, int *dest, long source_start, long source_end, long dest_start);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    long og_arr_size = atof(argv[1]);
    long k = log2(og_arr_size); //number of elements in each partion
    long r = ceil((double) og_arr_size/k); //number of partions
    //master's first partition may have less elements if doesn't divide evenly
    long num_elem_in_first_part = (rank == MASTER_PROC) ? (og_arr_size % r) ? k : og_arr_size % r : k;
    //master may have more partitions if doesn't divide evenly
    long num_proc_parts = (rank == MASTER_PROC) ? r/num_procs + r%num_procs : r/num_procs;

    //vars for MPI_Scatterv
    int *j_arr, *a_arr, *b_arr, *send_counts, *displs;

    send_counts = (int *) malloc(sizeof(int) * num_procs);
    displs = (int *) malloc(sizeof(int) * num_procs);

    //set first values
    send_counts[0] = num_proc_parts;
    displs[0] = 0;

    for (int i = 1; i < num_procs; i++) {
        //+1, -1 b/c need previous j value to calculate size of received array
        send_counts[i] = num_proc_parts + 1;
        displs[i] = send_counts[i - 1] + displs[i - 1] - 1;
    }

    //master create and sort 2 arrays of random numbers with size og_arr_size
    //then find J set
    if (rank == MASTER_PROC) {
        srand(time(NULL));

        a_arr = (int *) malloc(sizeof(int) * og_arr_size);
        b_arr = (int *) malloc(sizeof(int) * og_arr_size);
        fill_and_sort(a_arr, og_arr_size);
        fill_and_sort(b_arr, og_arr_size);

        j_arr = (int *) malloc(sizeof(int) * r);
        find_j_set(a_arr, b_arr, og_arr_size, j_arr, r, k, num_elem_in_first_part);
    } else {
        //+1 b/c need previous j value to calculate size of received array
        j_arr = (int *) malloc(sizeof(int) * num_proc_parts + 1);
    }

    //master distributes J set
    //+1 is for procs (other than master) to receive previous j element
    MPI_Scatterv(j_arr, send_counts, displs, MPI_INT, j_arr, num_proc_parts + 1, MPI_INT, MASTER_PROC,
                 MPI_COMM_WORLD);

    //each proc finds size of its B partition. Need to declare outside to avoid compiler warning
    int b_size;
    if (rank == MASTER_PROC) {
        b_size = find_b_size(j_arr, 0, num_proc_parts);
    } else {
        //pass start=1 b/c 0 is not part of proc's assigned set
        b_size = find_b_size(j_arr, 1, num_proc_parts + 1);
    }

    //allocate A and B arrs
    if (rank != MASTER_PROC) {
        a_arr = (int *) malloc(sizeof(int) * k * num_proc_parts);
        b_arr = (int *) malloc(sizeof(int) * b_size);
    }

    //master gets all B sizes, which is value of B to send to each proc
    MPI_Gather(&b_size, 1, MPI_INT, send_counts, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

    int *recv_counts = NULL;
    //save b sizes for later use as recv_counts
    if (rank == MASTER_PROC) {
        recv_counts = (int *) malloc(sizeof(int) * num_procs);
        append_array(send_counts, recv_counts, 0, num_procs - 1, 0);
    }

    //displs[0] already set from the last Scatterv
    if (rank == MASTER_PROC) {
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    }  

    //master distributes B array
    MPI_Scatterv(b_arr, send_counts, displs, MPI_INT, b_arr, b_size, MPI_INT, MASTER_PROC,
                 MPI_COMM_WORLD);

    //calculate send_counts to be the number of A elements each proc receives
    if (rank == MASTER_PROC) {
        send_counts[0] = num_elem_in_first_part + (num_proc_parts - 1) * k;
        //displs[0] already set

        for (int i = 1; i < num_procs; i++) {
            send_counts[i] = k * num_proc_parts;
            displs[i] = send_counts[i - 1] + displs[i - 1];
        }
    }

    //master distributes A array
    MPI_Scatterv(a_arr, send_counts, displs, MPI_INT, a_arr, num_elem_in_first_part + (num_proc_parts - 1) * k,
                 MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

    //wait for all processes to start before beginning for timing reasons
    //declare start_time for all procs to avoid compiler warning later
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int *c_arr;
    int c_size = num_elem_in_first_part + k * (num_proc_parts - 1) + b_size;

    if (rank == MASTER_PROC) {
        c_arr = (int *) malloc(sizeof(int) * og_arr_size * 2); //master needs to be able to store entire C array
    } else {
        c_arr = (int *) malloc(sizeof(int) * c_size);
    }

    merge_arrays(a_arr, b_arr, c_arr, j_arr, num_proc_parts, k);
    
    //calculate counts for gathering C arrays into master C arr
    if (rank == MASTER_PROC) {
        recv_counts[0] += c_size;
        //displs[0] already set

        for (int i = 1; i < num_procs; i++) {
            recv_counts[i] += k * num_proc_parts;
            displs[i] = recv_counts[i - 1] + displs[i - 1];
        }
    }
    
    //gather procs' C arrays into master
    //will be ordered in recv array (master's c_arr) according to rank, so will be in order
    MPI_Gatherv(c_arr, c_size, MPI_INT, c_arr, recv_counts, displs, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

    if (rank == MASTER_PROC) {
        double end_time = MPI_Wtime();
        double elapsed_seconds = end_time - start_time;

        printf("Took %.2f seconds\n\n", elapsed_seconds);

        //output C
        for (long i = 0; i < og_arr_size * 2; i++) {
            printf("%ld\n", c_arr[i]);
        }

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
            if (b_arr[mid] <= a_arr[num_elem_in_first_part + k * j - 1]) {
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

void merge_arrays(int *a_arr, int *b_arr, int *c_arr, int *j_arr, long num_proc_parts, long k, long num_elem_in_first_part) {
    long og_arr_size = num_elem_in_first_part + num_proc_parts * (k - 1);
    long a_index = 0, b_index = 0, c_index = 0;

    for (long current_part = 0; current_part < num_proc_parts; current_part++) {
        //if no more b partitions, merge all A into C
        if (j_arr[current_part] == -1) {
            append_array(a_arr, c_arr, a_index, og_arr_size - 1, c_index);
            break;
        }

        long a_part_end = num_elem_in_first_part + (k * current_part) - 1;

        //know that a_index won't go out of bounds because of how b is partitioned
        while (b_index <= j_arr[current_part]) {
            if (a_arr[a_index] <= b_arr[b_index]) {
                c_arr[c_index] = a_arr[a_index];
                a_index++;
            } else {
                c_arr[c_index] = b_arr[b_index];
                b_index++;
            }
            
            c_index++;
        }

        //merge all remaining a part. elements
        if (a_index != a_part_end) {
            append_array(a_arr, c_arr, a_index, a_part_end, c_index);
            a_index = a_part_end + 1;
        }
    }

    //if no b partitions, all A < all B and master has to append B to C (no other procs have B elements in this case)
    if (rank == MASTER_PROC && j_arr[0] == -1) {
        append_array(b_arr, c_arr, 0, og_arr_size - 1, og_arr_size);
    }    
}

void append_array(int *source, int *dest, long source_start, long source_end, long dest_start) {
    memcpy(dest[dest_start], source[source_start], sizeof(int) * (source_end - source_start));
}
