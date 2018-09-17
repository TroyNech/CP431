/* 
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: September 17, 2018
* 
* Usage: mpirun -np num_procs a1 range_limit
* Description: Finds the largest difference between 2 consecutive primes, within the range 0-$limit
* Limitations: 
*   Assumes that there is at least 1 prime in each proc's range
*   Assumes that the range can be evenly divided among the procs
*/

#include <gmp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER_PROC 0

typedef struct proc_diff_result {
    double prime1, prime2, diff, first_prime, last_prime;
} proc_diff_result;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argc);

    clock_t start_time = clock();

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double global_range_limit, range_limit;
    mpz_t range_start, first_prime, last_prime, current_prime, previous_prime, current_diff, max_diff, max_diff_prime1, max_diff_prime2;
    mpz_inits(range_start, first_prime, last_prime, current_prime, previous_prime, current_diff, max_diff, max_diff_prime1, max_diff_prime2, NULL);

    global_range_limit = atof(argv[0]);

    //divide range among procs
    mpz_set_d(range_start, (global_range_limit/num_procs) * rank);
    range_limit = (global_range_limit/num_procs) * (rank + 1);

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
    proc_diff_result proc_diff_result, *global_proc_diff_results;
    proc_diff_result->prime1 = mpz_get_d(max_diff_prime1);
    proc_diff_result->prime2 = mpz_get_d(max_diff_prime2);
    proc_diff_result->diff = mpz_get_d(max_diff);
    proc_diff_result->first_prime = mpz_get_d(first_prime);
    proc_diff_result->last_prime = mpz_get_d(last_prime);

    mpz_clear(range_start, first_prime, last_prime, current_prime, previous_prime, current_diff, max_diff, max_diff_prime1, max_diff_prime2, NULL);

    //only master proc needs recv var
    if (rank == MASTER_PROC) {
        global_proc_diff_results = (proc_diff_result *) malloc(sizeof(proc_diff_result)*num_procs);
    }

    MPI_Gather(&proc_diff_result, 5, MPI_DOUBLE, &global_proc_diff_results, 5, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);

    //find max diff of all procs
    if (rank == MASTER_PROC) {
        double global_max_prime1, global_max_prime2, edge_diff, global_max_diff = 0;

        for (int i = 0; i < num_procs-1; i++) {
            edge_diff = global_proc_diff_results[i+1]->first_prime - global_proc_diff_results[i]->last_prime;
            if (edge_diff > global_proc_diff_results[i]->diff) {
                if (edge_diff > global_max_diff) {
                    global_max_diff = edge_diff;
                    global_max_prime1 = global_proc_diff_results[i+1]->first_prime;
                    global_max_prime2 = global_proc_diff_results[i]->last_prime;
                } 
            } else if (global_proc_diff_results[i]->diff > global_max_diff) {
                global_max_diff = global_proc_diff_results[i]->diff;
                global_max_prime1 = global_proc_diff_results[i]->prime1;
                global_max_prime2 = global_proc_diff_results[i]->prime2;
            }
        }

        //check last proc's max diff
        if (global_proc_diff_results[num_procs-1]]->diff > global_max_diff) {
            global_max_diff = global_proc_diff_results[num_procs-1]->diff;
            global_max_prime1 = global_proc_diff_results[num_procs-1]->prime1;
            global_max_prime2 = global_proc_diff_results[num_procs-1]->prime2;
        }

        printf("Maximum difference in range 1 - %f is %f between %f and %f", global_range_limit, global_max_diff, global_max_prime1, global_max_prime2);
    }
    
    clock_t end_time = clock();
    float elapsed_seconds = float(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Process %d took %f.3 seconds", rank, elapsed_seconds);

    MPI_Finalize();

    return EXIT_SUCCESS;
}