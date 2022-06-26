#define _GNU_SOURCE

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "sched.h"
#include "omp.h"

double drand(double low, double high, unsigned int *seed)
{
    return ((double)rand_r(seed) * (high - low)) / (double)RAND_MAX + low;
}

void init_vars(int *N, int *M, int *mode)
{
    const char *number_of_instances = getenv("N");
    if (number_of_instances)
    {
        *N = atoi(number_of_instances);
    }

    const char *matrix_dim = getenv("M");
    if (matrix_dim)
    {
        *M = atoi(matrix_dim);
    }

    const char *running_mode = getenv("MODE");
    if (running_mode)
    {
        *mode = atoi(running_mode);
    }
}

void init_matrix(double *instance, unsigned int M)
{
    unsigned int seed = 25234 + 17*omp_get_thread_num();
    for (int row = 0; row < M; row++)
    {
        for (int column = 0; column <= M; column++)
        {
            instance[row * (M + 1) + column] = drand(-1.0, 1.0, &seed);
        }
    }
}

void print_matrix(double *instance, int M)
{
    for (int row = 0; row < M; row++)
    {
        for (int column = 0; column <= M; column++)
        {
            printf("%lf ", instance[row * (M + 1) + column]);
        }
        printf("\n");
    }
    printf("\n");
}

void swap(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

void initialize_and_solve(int M)
{
    double *instance;
    instance = malloc(M * (M + 1) * sizeof(double));

    int i, j, k = 0, c, flag = 0, m = 0;
    
    // Initialise the matrix
    init_matrix(instance, M);
    
    // Performing elementary operations
    for (i = 0; i < M; i++)
    {
        if (instance[i * (M + 2)] == 0)
        {
            c = 1;
            int idx = (i + c) * (M + 1) + i;
            while ((i + c) < M && instance[idx] == 0) {
                c++;
                idx = (i + c) * (M + 1) + i;
            }

            if ((i + c) == M)
            {
                flag = 1;
                break;
            }

            for (j = i, k = 0; k <= M; k++) {
                int lhs = j * (M + 1) + k;
                int rhs = (j + c) * (M + 1) + k;
                swap(&instance[lhs], &instance[rhs]);
            }

        }

        for (j = 0; j < M; j++)
        {
            // Excluding all i == j
            if (i != j)
            {
                // Converting Matrix to reduced row
                // echelon form(diagonal matrix)
                int idx_num = j * (M + 1) + i;
                int idx_den = i * (M + 2);
                double pro = instance[idx_num] / instance[idx_den];

                for (k = 0; k <= M; k++) {
                    int idx1 = j * (M + 1) + k;
                    int idx2 = i * (M + 1) + k;
                    instance[idx1] -= (instance[idx2]) * pro;
                }
            }
        }
    }

    // Get the solution in the last column
    for (int i = 0; i < M; i++)
    {
        instance[i * (M + 1) + M] /= instance[i * (M + 2)];
    }

    free(instance); instance = NULL;
}

double solve_serial(int N, int M)
{
    double now = omp_get_wtime();
    for (int i = 0; i < N; i++)
    {
        initialize_and_solve(M);
    }
    return omp_get_wtime() - now;
}

double solve_parallel(int N, int M)
{
    double now = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        initialize_and_solve(M);
    }
    return omp_get_wtime() - now;
}

int main(int argc, char **argv)
{
    // Print CPU information
    // int status = system("lscpu -e");

    // Default parameters
    int N = 200, M = 200, mode = 2;

    if (argc != 4)
    {
        printf("Not enough arguments or too many arguments passed\n");
        exit(0);
    }
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    mode = atoi(argv[3]);

    init_vars(&N, &M, &mode);

    if (mode == 0)
    {
        // Serial only
        double l2_norm_serial = 0.0;
        double serial = solve_serial(N, M);
        printf("Time, %d, %d, %lf\n", N, M, serial);
    }
    else if (mode == 1)
    {
        // Parallel only
        double l2_norm_parallel = 0.0;
        double parallel = solve_parallel(N, M);
        printf("Time, %d, %d, %lf\n", N, M, parallel);
    }
    else
    {
        // Both serial and parallel
        // Solve using GEM (serial)
        double serial = solve_serial(N, M);

        // Solve using GEM (parallel)
        double parallel = solve_parallel(N, M);

        printf("Time, %d, %d, %lf, %lf, %lf\n", N, M, serial, parallel, serial / parallel);
    }

    return 0;
}
