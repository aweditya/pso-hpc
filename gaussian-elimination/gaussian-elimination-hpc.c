#define _GNU_SOURCE

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

void init_vars(int *N, int *M)
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
}

void init_matrices(double *a, double *b, int N, int M)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            for (int k = 0; k <= M; k++)
            {
                a[i * M * (M + 1) + j * (M + 1) + k] = drand(-1.0, 1.0);
                b[i * M * (M + 1) + j * (M + 1) + k] = a[i * M * (M + 1) + j * (M + 1) + k];
            }
        }
    }
}

void print_matrix(double *a, int N, int M)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            for (int k = 0; k <= M; k++)
            {
                printf("%lf ", a[i * M * (M + 1) + j * (M + 1) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void swap(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

void solve_one_instance(double *instance, int N, int M)
{
    int i, j, k = 0, c, flag = 0, m = 0;

    // Performing elementary operations
    for (i = 0; i < M; i++)
    {
        if (instance[i * (M + 2)] == 0)
        {
            c = 1;
            while ((i + c) < M && instance[(i + c) * (M + 1) + i] == 0)
                c++;

            if ((i + c) == M)
            {
                flag = 1;
                break;
            }

            for (j = i, k = 0; k <= M; k++)
                swap(&instance[j * (M + 1) + k], &instance[(j + c) * (M + 1) + k]);
        }

        for (j = 0; j < M; j++)
        {
            // Excluding all i == j
            if (i != j)
            {
                // Converting Matrix to reduced row
                // echelon form(diagonal matrix)
                double pro = instance[j * (M + 1) + i] / instance[i * (M + 2)];

                for (k = 0; k <= M; k++)
                    instance[j * (M + 1) + k] -= (instance[i * (M + 1) + k]) * pro;
            }
        }
    }

    // Get the solution in the last column
    for (int i = 0; i < M; i++)
        instance[i * (M + 1) + M] /= instance[i * (M + 2)];
}

double solve_serial(double *a, int N, int M)
{
    double now = omp_get_wtime();
    for (int i = 0; i < N; i++)
    {
        solve_one_instance(a + i * M * (M + 1), N, M);
    }
    return omp_get_wtime() - now;
}

double solve_parallel(double *a, int N, int M)
{
    double now = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        solve_one_instance(a + i * M * (M + 1), N, M);
    }
    return omp_get_wtime() - now;
}

// L2 norm of solution (last column of instance)
double l2_norm(double *instance, int N, int M)
{
    double norm = 0.0;
    for (int i = 0; i < M; i++)
    {
        norm += instance[i * (M + 1) + M] * instance[i * (M + 1) + M];
    }

    return norm;
}

double sum_l2(double *a, int N, int M)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum += l2_norm(a + i * M * (M + 1), N, M);
    }

    return sum;
}

int main(int argc, char **argv)
{
    // Default parameters
    int N = 200, M = 200;

    if (argc == 3)
    {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
    }

    init_vars(&N, &M);

    double *a, *b;
    a = malloc(N * M * (M + 1) * sizeof(double));
    b = malloc(N * M * (M + 1) * sizeof(double));

    // Random Initialisation
    init_matrices(a, b, N, M);

    // Solve using GEM (serial)
    double serial = solve_serial(a, N, M);
    printf("Sum of L2 norm: %lf\n", sum_l2(a, N, M));
    printf("Runtime (without multithreading): %lf\n", serial);

    // Solve using GEM (parallel)
    double parallel = solve_parallel(b, N, M);
    printf("Sum of L2 norm: %lf\n", sum_l2(b, N, M));
    printf("Runtime (with multithreading): %lf\n", parallel);

    free(a); a = NULL;
    free(b); b = NULL;
    return 0;
}
