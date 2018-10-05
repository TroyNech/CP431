/* 
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: October 4, 2018
* 
* Usage: num_procs a1 range_limit
* Description: Finds the largest difference between 2 consecutive primes, within the range 0-$limit
* Limitations: 
*   Assumes that there is at least 1 prime in each proc's range
*   Assumes that the input is correct
* Note: Needs to be compiled like 'mpicc a1.c -o a1.out $CPPFLAGS $LDFLAGS -lgmp -std=c99' on SHARCNET orca system
*/

#include <gmp.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER_PROC 0

typedef struct {
    double prime1, prime2, diff, first_prime, last_prime;
} proc_diff_result_struct;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    //wait for all processes to start before beginning for timing reasons
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double global_range_limit, range_limit, range_size;
    mpz_t range_start, first_prime, last_prime, current_prime, previous_prime, current_diff, max_diff, max_diff_prime1, max_diff_prime2;
    mpz_inits(range_start, first_prime, last_prime, current_prime, previous_prime, current_diff, max_diff, max_diff_prime1, max_diff_prime2, NULL);

    global_range_limit = atof(argv[1]);
    range_size = floor(global_range_limit/num_procs);

    //divide range among procs
    mpz_set_d(range_start, range_size * rank);
    range_limit = (rank == num_procs - 1) ? global_range_limit : range_size * (rank + 1);

    mpz_nextprime(current_prime, range_start);
    mpz_set(previous_prime, current_prime);
    mpz_set(first_prime, current_prime);

    //find max diff between consecutive procs
    while (mpz_cmp_d(current_prime, range_limit) <= 0) {
        mpz_sub(current_diff, current_prime, previous_prime);
        
        if (mpz_cmp(current_diff, max_diff) > 0) {
            mpz_set(max_diff, current_diff);
            mpz_set(max_diff_prime1, previous_prime);
            mpz_set(max_diff_prime2, current_prime);
        }

        mpz_set(previous_prime, current_prime);
        mpz_nextprime(current_prime, previous_prime);
    }

    mpz_set(last_prime, previous_prime);

    //create struct of doubles for reporting results to master proc
    //sending mpz_t would have required more work, would need to define type for MPI
    proc_diff_result_struct proc_diff_result, *global_proc_diff_results = NULL;
    proc_diff_result.prime1 = mpz_get_d(max_diff_prime1);
    proc_diff_result.prime2 = mpz_get_d(max_diff_prime2);
    proc_diff_result.diff = mpz_get_d(max_diff);
    proc_diff_result.first_prime = mpz_get_d(first_prime);
    proc_diff_result.last_prime = mpz_get_d(last_prime);
    
    mpz_clears(range_start, first_prime, last_prime, current_prime, previous_prime, current_diff, max_diff, max_diff_prime1, max_diff_prime2, NULL);

    //only master proc needs recv var
    if (rank == MASTER_PROC) {
        global_proc_diff_results = (proc_diff_result_struct *) malloc(sizeof(proc_diff_result_struct)*num_procs);
    }

   MPI_Gather(&proc_diff_result, 5, MPI_DOUBLE, global_proc_diff_results, 5, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);

    //find max diff of all procs
    if (rank == MASTER_PROC) {
        double global_max_prime1, global_max_prime2, edge_diff, global_max_diff = 0;

        for (int i = 0; i < num_procs-1; i++) {
            edge_diff = global_proc_diff_results[i+1].first_prime - global_proc_diff_results[i].last_prime;
            if (edge_diff > global_proc_diff_results[i].diff) {
                if (edge_diff > global_max_diff) {
                    global_max_diff = edge_diff;
                    global_max_prime1 = global_proc_diff_results[i+1].first_prime;
                    global_max_prime2 = global_proc_diff_results[i].last_prime;
                } 
            } else if (global_proc_diff_results[i].diff > global_max_diff) {
                global_max_diff = global_proc_diff_results[i].diff;
                global_max_prime1 = global_proc_diff_results[i].prime1;
                global_max_prime2 = global_proc_diff_results[i].prime2;
            }
        }

        //check last proc's max diff
        if (global_proc_diff_results[num_procs-1].diff > global_max_diff) {
            global_max_diff = global_proc_diff_results[num_procs-1].diff;
            global_max_prime1 = global_proc_diff_results[num_procs-1].prime1;
            global_max_prime2 = global_proc_diff_results[num_procs-1].prime2;
        }

        printf("Maximum difference in range 1 - %0.f is %.0f between %.0f and %.0f\n", global_range_limit, global_max_diff, global_max_prime1, global_max_prime2);

    double end_time = MPI_Wtime();
    double elapsed_seconds = end_time - start_time;

    printf("Took %.2f seconds\n", elapsed_seconds);

    }
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}
