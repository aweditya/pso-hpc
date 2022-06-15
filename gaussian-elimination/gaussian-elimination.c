#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#define M 2

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

void init_matrix(double *a)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j <= M; j++)
        {
            a[i * (M + 1) + j] = drand(1.0, 10.0);
        }
    }
}

void print_matrix(double *a)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j <= M; j++)
        {
            printf("%lf ", a[i * (M + 1) + j]);
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

void solve(double *a)
{
    int i, j, k = 0, c, flag = 0, m = 0;

    // Performing elementary operations
    for (i = 0; i < M; i++)
    {
        if (a[i * (M + 2)] == 0)
        {
            c = 1;
            while ((i + c) < M && a[(i + c) * (M + 1) + i] == 0)
                c++;

            if ((i + c) == M)
            {
                flag = 1;
                break;
            }

            for (j = i, k = 0; k <= M; k++)
                swap(&a[j * (M + 1) + k], &a[(j + c) * (M + 1) + k]);
        }

        for (j = 0; j < M; j++)
        {
            // Excluding all i == j
            if (i != j)
            {
                // Converting Matrix to reduced row
                // echelon form(diagonal matrix)
                double pro = a[j * (M + 1) + i] / a[i * (M + 2)];

                for (k = 0; k <= M; k++)
                    a[j * (M + 1) + k] -= (a[i * (M + 1) + k]) * pro;
            }
        }
    }

    // Get the solution in the last column
    for (int i = 0; i < M; i++)
        a[i * (M + 1) + M] /= a[i * (M + 2)];
}

// L2 norm of solution (last column of a)
double l2_norm(double *a)
{
    double norm = 0.0;
    for (int i = 0; i < M; i++)
    {
        norm += a[i * (M + 1) + M] * a[i * (M + 1) + M];
    }

    return norm;
}

int main()
{
    double *a;
    a = malloc(M * (M + 1) * sizeof(double));

    init_matrix(a);

    solve(a);
    printf("%lf\n", l2_norm(a));

    free(a); a = NULL;
    return 0;
}
