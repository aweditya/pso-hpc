#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"
#include "sched.h"

#define number 200000
#define length 10

double drand(double low, double high)
{
        return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

void sort(double* A)
{
        double temp;
        for (int i = 0; i < length - 1; i++)
        {
                for (int j = 0; j < length - i - 1; j++)
                {
                        if (A[j] > A[j + 1]) 
                        {
                                temp = A[j];
                                A[j] = A[j + 1]; 
                                A[j + 1] = temp;
                        }
                }
        }
}

int main(int argc, char **argv)
{
        // srand(time(0));
        // int thread_count = atoi(argv[1]);

        double *A; 
        A = malloc(number * length * sizeof(double));
    
        for (int i = 0; i < number * length; i++)
        {
                *(A + i) = drand(0.0, 1.0);
        }

        // omp_set_num_threads(thread_count);
        double now = omp_get_wtime();
#pragma omp parallel
{
        printf("Number of threads: %d\n", omp_get_num_threads());
#pragma omp for 
        for (int j = 0; j < number; j++)
        {    
                sort(A + length * j); 
        }
}
        double multithreading = omp_get_wtime() - now;
        printf("Execution time for Bubble sort on %d arrays of length %d: %f\n", number, length, multithreading);

        free(A); A = NULL;
        return 0;
}
