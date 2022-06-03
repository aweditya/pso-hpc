#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "sched.h"

double drand(double low, double high)
{
	return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
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

int main(int argc, char **argv)
{
	int number = 100000;
	int length = 10;
	int thread_number = omp_get_max_threads();
	int mode = 2;

	if (argc == 5)
	{
		thread_number = atoi(argv[1]);
		number = atoi(argv[2]);
		length = atoi(argv[3]);
		mode = atoi(argv[4]);
	}

	double *A, *B;
	A = malloc(number * length * sizeof(double));
	B = malloc(number * length * sizeof(double));

	init_array(A, number, length);
	init_array(B, number, length);

	double now;
	double no_multithreading, multithreading;

	now = omp_get_wtime();
	for (int j = 0; j < number; j++)
	{
		sort(A + length * j, length);
	}
	no_multithreading = omp_get_wtime() - now;

	omp_set_num_threads(thread_number);
	now = omp_get_wtime();
#pragma omp parallel for
	for (int j = 0; j < number; j++)
	{
		// printf("%d\n", sched_getcpu());
		sort(B + length * j, length);
	}
	multithreading = omp_get_wtime() - now;

	if (mode == 0)
	{
		printf("%lf %d %d %d 1 %lf\n", omp_get_wtime(), mode, number, length, no_multithreading);
	}
	else if (mode == 1)
	{
		printf("%lf %d %d %d %d %lf\n", omp_get_wtime(), mode, number, length, thread_number, multithreading);
	}
	else if (mode == 2)
	{
		printf("%lf %d %d %d %d %lf %lf\n", omp_get_wtime(), mode, number, length, thread_number, no_multithreading, multithreading);
	}
	else
	{
		printf("Mode not supported\n");
	}

	free(A);
	A = NULL;
	free(B);
	B = NULL;
	return 0;
}
