#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include "sched.h"

double drand(double low, double high)
{
        return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

void init_vars(int *number, int *length, int *thread_number, int *mode)
{
        const char *arr_dim = getenv("ARR_DIM");
        if (arr_dim)
        {
                *number = atoi(arr_dim);
        }

        const char *chunk_len = getenv("CHUNK_LEN");
        if (chunk_len)
        {
                *length = atoi(chunk_len);
        }

        const char *thread_count = getenv("THREAD_COUNT");
        if (thread_count)
        {
                *thread_number = atoi(thread_count);
        }

        const char *run_mode = getenv("RUN_MODE");
        if (run_mode)
        {
                *mode = atoi(run_mode);
        }
}

void init_array(double *A, int number, int length)
{
        for (int i = 0; i < number * length; i++)
        {
                *(A + i) = drand(0.0, 1.0);
        }
}

void sort(double *A, int length)
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

double run_sort_single(int number, int length)
{
        double *A;
        A = malloc(number * length * sizeof(double));
        init_array(A, number, length);

        double now = omp_get_wtime();
        for (int j = 0; j < number; j++)
        {
                sort(A + length * j, length);
        }
        double run_time = omp_get_wtime() - now;
        free(A);

	return run_time;
}

double run_sort_parallel(int number, int length, int thread_number)
{
        double *A = malloc(number * length * sizeof(double));
        init_array(A, number, length);

        omp_set_num_threads(thread_number);

        double now = omp_get_wtime();
#pragma omp parallel for
        for (int j = 0; j < number; j++)
        {
                sort(A + length * j, length);
        }
        double run_time = omp_get_wtime() - now;
        free(A);

	return run_time;
}

/*
Execute using
aprun -n 1 -N 1 -d 40 -e ARR_DIM=1000000 -e CHUNK_LEN=10 -e THREAD_COUNT=40 -e RUN_MODE=2 ./bubble_sort
*/
int main(int argc, char **argv)
{
        time_t rawtime;
        struct tm *timeinfo;

        time(&rawtime);
        timeinfo = localtime(&rawtime);
        printf("%s", asctime(timeinfo));

        int number = 100000;
        int length = 10;
        int thread_number = omp_get_max_threads();
        int mode = 2;

        init_vars(&number, &length, &thread_number, &mode);

        if (argc == 5)
        {
                thread_number = atoi(argv[1]);
                number = atoi(argv[2]);
                length = atoi(argv[3]);
                mode = atoi(argv[4]);
        }

        if (mode == 0)
        {
                double no_multithreading = run_sort_single(number, length);
                printf("%d %d %d 1 %lf\n", mode, number, length, no_multithreading);
        }
        else if (mode == 1)
        {
                double multithreading = run_sort_parallel(number, length, thread_number);
                printf("%d %d %d %d %lf\n", mode, number, length, thread_number, multithreading);
        }
        else if (mode == 2)
        {
                double no_multithreading = run_sort_single(number, length);
                double multithreading = run_sort_parallel(number, length, thread_number);
                printf("%d %d %d %d %lf %lf\n", mode, number, length, thread_number, no_multithreading, multithreading);
        }
        else
        {
                printf("Mode not supported\n");
        }
}
