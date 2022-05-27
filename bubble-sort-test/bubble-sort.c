#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define length 1000000

double drand(double low, double high)
{
        return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

int main()
{
	double *A;
	A = malloc(length * sizeof(double));

	for (int i = 0; i < length; i++)
	{
		*(A + i) = drand(0.0, 1.0); 
	}

	double temp;
	double now = omp_get_wtime();
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
	double time = omp_get_wtime() - now;
	printf("Sorting time for array of length %d: %f\n", length, time);

	free(A); A = NULL;
	return 0;
}
