#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#define N 4
#define M 2

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

void init_matrix(double *a)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            for (int k = 0; k <= M; k++)
            {
                a[i * M * (M + 1) + j * (M + 1) + k] = drand(1.0, 10.0);
            }
        }
    }
}

void print_matrix(double *a)
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

void solve_one_instance(double *instance)
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

void solve_serial(double *a)
{
    for (int i = 0; i < N; i++)
    {
        solve_one_instance(a + i * M * (M + 1));
    }
}

// L2 norm of solution (last column of instance)
double l2_norm(double *instance)
{
    double norm = 0.0;
    for (int i = 0; i < M; i++)
    {
        norm += instance[i * (M + 1) + M] * instance[i * (M + 1) + M];
    }

    return norm;
}

double sum_l2(double *a)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum += l2_norm(a + i * M * (M + 1));
    }

    return sum;
}

int main()
{
    double *a;
    a = malloc(N * M * (M + 1) * sizeof(double));

    // Random Initialisation
    init_matrix(a);
 
    // Solve using GEM (serial)
    solve_serial(a);
    printf("Sum of L2 norm: %lf\n", sum_l2(a));
    
    return 0;
}