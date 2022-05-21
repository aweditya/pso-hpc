#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define dimA 80000
#define dimB 10000

double drand(double low, double high)
{
	return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

int main()
{
	int i;
	double *A, *B, *C;

	A = malloc(dimA * sizeof(double));
	B = malloc(dimB * sizeof(double));
	C = malloc(dimA * dimB * sizeof(double));

	for (i = 0; i < dimA; i++)
	{
		*(A + i) = drand(0.0, 1.0);
	}
	for (i = 0; i < dimB; i++)
	{
		*(B + i) = drand(0.0, 1.0);
	}

	// No multithreading
	int p, q;
	double now = omp_get_wtime();
	for (p = 0; p < dimA; p++)
	{
		for (q = 0; q < dimB; q++)
		{
			C[dimB * p + q] = A[p] * B[q];
		}
	}
	double no_multithreading = omp_get_wtime() - now;

	// Multithreading
	now = omp_get_wtime();
#pragma omp parallel for collapse(2)
	for (int j = 0; j < dimA; j++)
	{
		for (int k = 0; k < dimB; k++)
		{
			C[dimB * j + k] = A[j] * B[k];
		}
	}
	double multithreading = omp_get_wtime() - now;
	printf("Speedup for dimA: %d and dimB: %d: %f\n", dimA, dimB, no_multithreading / multithreading);

	free(A);
	A = NULL;

	free(B);
	B = NULL;

	free(C);
	C = NULL;
}
