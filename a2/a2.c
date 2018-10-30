/*
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: October 31, 2018
*
* Usage: a2 size
* Description:
*   Generates 2 arrays of random numbers with size $size, sorts them, then merges them in parallel
*   Merges using algorithm detailed at https://mylearningspace.wlu.ca/d2l/le/content/285513/viewContent/1586397/View
*   Reports time taken to merge and outputs arrays (to stdout)
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
void find_j_set(int *a_arr, int *b_arr, long og_arr_size, long *j_arr, long r, long k, long num_elem_in_first_part);
int find_b_size(long *j_arr, long start, long end, int rank);
void merge_arrays(int *a_arr, int *b_arr, int *c_arr, long *j_arr, long num_proc_parts, long k, long num_elem_in_first_part,
            int rank);
void append_array(int *source, int *dest, long source_start, long source_end, long dest_start);
void output_arr(int *arr, long size, char *msg);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    long og_arr_size = atof(argv[1]);
    long k = log2(og_arr_size); //number of elements in each partion
    long r = ceil((double) og_arr_size / k); //number of partions
    //master may have more partitions if doesn't divide evenly
    long num_proc_parts = (rank == MASTER_PROC) ? r / num_procs + r % num_procs : r / num_procs;
    long num_elem_in_first_part = k;

    
    //declare outside to avoid compiler error
    long num_other_proc_parts;
    if (rank == MASTER_PROC) {
        //master needs to know how many partitions the other procs
        num_other_proc_parts = r / num_procs;

        //master's first partition may have less elements if doesn't divide evenly
        if (og_arr_size % k != 0) {
            num_elem_in_first_part = og_arr_size % k;
        }
    }

    //vars for MPI_Scatterv
    int *a_arr, *b_arr, *send_counts, *displs;
    long *j_arr;

    if (rank == MASTER_PROC) {
        send_counts = (int *) malloc(sizeof(int) * num_procs);
        displs = (int *) malloc(sizeof(int) * num_procs);

        //set first values
        send_counts[0] = num_proc_parts;
        displs[0] = 0;

        for (int i = 1; i < num_procs; i++) {
            //+1, -1 b/c need previous j value to calculate size of received array
            send_counts[i] = num_other_proc_parts + 1;
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

        j_arr = (long *) malloc(sizeof(long) * r);
        find_j_set(a_arr, b_arr, og_arr_size, j_arr, r, k, num_elem_in_first_part);
    } else {
        //+1 element b/c need previous valid j value to calculate size of received array
        j_arr = (long *) malloc(sizeof(long) * num_proc_parts + sizeof(long));
    }

    //master distributes J set
    //MPI_IN_PLACE so that master doesn't send to itself (avoids error of using j_arr as send and recv buffer)
    if (rank == MASTER_PROC) {
        MPI_Scatterv(j_arr, send_counts, displs, MPI_LONG, MPI_IN_PLACE, num_proc_parts, MPI_LONG, MASTER_PROC,
                    MPI_COMM_WORLD);
    }
    //+1 is for procs to receive previous valid j value 
    else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_LONG, j_arr, num_proc_parts + 1, MPI_LONG, MASTER_PROC,
                    MPI_COMM_WORLD);
    }  

    //each proc finds size of its B partition. Need to declare outside to avoid compiler warning
    int b_size;
    if (rank == MASTER_PROC) {
        b_size = find_b_size(j_arr, 0, num_proc_parts - 1, rank);
    } else {
        //pass start=1 b/c 0 is not part of proc's assigned set
        b_size = find_b_size(j_arr, 1, num_proc_parts, rank);
    }

    //allocate A and B arrs
    if (rank != MASTER_PROC) {
        a_arr = (int *) malloc(sizeof(int) * k * num_proc_parts);
        b_arr = (int *) malloc(sizeof(int) * b_size);
    }
    
    if (rank == MASTER_PROC) {
        //calculate send_counts to be the number of A elements each proc receives
        send_counts[0] = num_elem_in_first_part + (num_proc_parts - 1) * k;
        //displs[0] already set

        for (int i = 1; i < num_procs; i++) {
            send_counts[i] = k * num_other_proc_parts;
            displs[i] = send_counts[i - 1] + displs[i - 1];
        }

        //master distributes A array
        //MPI_IN_PLACE so that master doesn't send to itself (avoids error of using a_arr as send and recv buffer)
        MPI_Scatterv(a_arr, send_counts, displs, MPI_INT, MPI_IN_PLACE, num_elem_in_first_part + (num_proc_parts - 1) * k,
                    MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, send_counts, displs, MPI_INT, a_arr, num_elem_in_first_part + (num_proc_parts - 1) * k,
                    MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
    }

    //master gets all B sizes, which is value of B to send to each proc
    MPI_Gather(&b_size, 1, MPI_INT, send_counts, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

    int *recv_counts = NULL;

    if (rank == MASTER_PROC) {
        //save b sizes for later use as recv_counts
        recv_counts = (int *) malloc(sizeof(int) * num_procs);
        append_array(send_counts, recv_counts, 0, num_procs - 1, 0);

        //set displs
        //displs[0] already set from the last Scatterv
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    
        //master distributes B array
        //MPI_IN_PLACE so that master doesn't send to itself (avoids error of using b_arr as send and recv buffer)
        MPI_Scatterv(b_arr, send_counts, displs, MPI_INT, MPI_IN_PLACE, b_size, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, b_arr, b_size, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
    }
    
    //wait for all processes to start before beginning for timing reasons
    //declare start_time for all procs to avoid compiler warning later
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int *c_arr;
    int c_size = num_elem_in_first_part + k * (num_proc_parts - 1) + b_size;

    if (rank == MASTER_PROC) {
        //send_counts no longer used
        free(send_counts);

        //master needs to be able to store entire C array
        c_arr = (int *) malloc(sizeof(int) * og_arr_size * 2);
    } else {
        c_arr = (int *) malloc(sizeof(int) * c_size);
    }

    merge_arrays(a_arr, b_arr, c_arr, j_arr, num_proc_parts, k, num_elem_in_first_part, rank);

    if (rank == MASTER_PROC) {
        //calculate counts for gathering C arrays into master C arr
        //recv_counts already set to b sizes
        //displs[0] already set
        recv_counts[0] += num_elem_in_first_part + k * (num_proc_parts - 1);

        for (int i = 1; i < num_procs; i++) {
            recv_counts[i] += k * num_other_proc_parts;
            displs[i] = recv_counts[i - 1] + displs[i - 1];
        }
    
        //gather procs' C arrays into master
        //will be ordered in recv array (master's c_arr) according to rank, so will be in order
        //MPI_IN_PLACE so that master doesn't send to itself (avoids error of using c_arr as send and recv buffer)
        MPI_Gatherv(MPI_IN_PLACE, c_size, MPI_INT, c_arr, recv_counts, displs, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(c_arr, c_size, MPI_INT, NULL, recv_counts, displs, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
    }

    if (rank == MASTER_PROC) {
        double end_time = MPI_Wtime();
        double elapsed_seconds = end_time - start_time;

        printf("\nTook %.2f seconds to merge arrays\n", elapsed_seconds);

        output_arr(a_arr, og_arr_size, "A:");
        output_arr(b_arr, og_arr_size, "B:");
        output_arr(c_arr, og_arr_size * 2, "C:");
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
    return *i - *j;
}

void find_j_set(int *a_arr, int *b_arr, long og_arr_size, long *j_arr, long r, long k, long num_elem_in_first_part) {
    long low = 0, prev_low = 0, last_valid_j = -1;
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

        //low increments if acceptable value found and so will not equal prev_low
        //will give last_valid_j in case that first available B value is > A[k*(j+1)]
        // or if all available B vals are < A[k*(j+1)]
        j_arr[j] = (low == prev_low) ? last_valid_j : low - 1;
        last_valid_j = j_arr[j];
        prev_low = low;
    }
}

int find_b_size(long *j_arr, long start, long end, int rank) {
    //find last valid j value
    while (end > start - 1 && j_arr[end] == -1) {
        end--;
    }

    //size is 0 if no B partitions, else equal to the number of elements between
    //use start var instead of 0 b/c start is 0 for master and 1 for others (element 0 not assigned to them)
    if (end == start - 1) {
        return 0;
    } 
    //master proc j values are relative to 0
    else if (rank == MASTER_PROC) {
        return j_arr[end] + 1;
    }
    //other proc j values are relative to last valid j value of previous procs (j_arr[0])
    else {
        return j_arr[end] - j_arr[0];
    }
}

void merge_arrays(int *a_arr, int *b_arr, int *c_arr, long *j_arr, long num_proc_parts, long k, long num_elem_in_first_part, int rank) {
    long a_size = num_elem_in_first_part + k * (num_proc_parts - 1);
    long a_index = 0, b_index = 0, c_index = 0;

    //first element for procs other than master is last j value of previous proc, so its j values start at j_arr[1]
    long j_index = (rank == MASTER_PROC) ? 0 : 1;

    for (long current_part = 0; current_part < num_proc_parts; current_part++, j_index++) {
        long a_part_end = num_elem_in_first_part + (k * current_part) - 1;
        //master j values always relative to 0, other procs' values relative to last j value of previous proc
        //  which they have as j_arr[0]
        long b_part_end = (rank == MASTER_PROC) ? j_arr[j_index] : j_arr[j_index] - j_arr[0] - 1;

        //know that a_index won't go out of bounds because of how b is partitioned
        while (b_index <= b_part_end) {
            if (a_arr[a_index] < b_arr[b_index]) {
                c_arr[c_index] = a_arr[a_index];
                a_index++;
            } else {
                c_arr[c_index] = b_arr[b_index];
                b_index++;
            }
            
            c_index++;
        }

        //merge all remaining a part. elements
        if (a_index <= a_part_end) {
            append_array(a_arr, c_arr, a_index, a_part_end, c_index);
            
            c_index += a_part_end - a_index + 1;
            a_index = a_part_end + 1;
        }
    }

    //master needs to merge B elements that were not partitioned (it has access to all of B and C)
    if (rank == MASTER_PROC) {
        //reconstruct variables declared at beginning of program rather than adding to function signature
        //  since they don't really make sense to pass and it keeps the signature shorter 
        long og_arr_size = exp2(k); //assume n=2^k
        long k = log2(og_arr_size); //number of elements in each partion
        long r = ceil((double) og_arr_size / k); //number of partions
        int total_b_size = find_b_size(j_arr, 0, r - 1, rank);

        append_array(b_arr, c_arr, total_b_size, og_arr_size - 1, og_arr_size + total_b_size);
    }    
}

void append_array(int *source, int *dest, long source_start, long source_end, long dest_start) {
    for (long i = 0; i <= source_end - source_start; i++) {
        dest[dest_start + i] = source[source_start + i];
    }
}

void output_arr(int *arr, long size, char *msg) {
    printf("\n%s\n", msg);
    
    for (long i = 0; i < size; i++) {
        printf("%d\n", arr[i]);
    }
}
