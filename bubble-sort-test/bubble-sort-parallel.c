#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define number 100000
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

int main()
{
	double *A, *B;
	A = malloc(number * length * sizeof(double));
	B = malloc(number * length * sizeof(double));
	
	for (int i = 0; i < number * length; i++)
	{
		*(A + i) = drand(0.0, 1.0); 
		*(B + i) = *(A + i);
	}

	/*
	for (int i = 0; i < number * length; i++)
	{
		printf("%f\n", A[i]);
	}
	*/
	double now = omp_get_wtime();
	for (int j = 0; j < number; j++)
	{
		sort(A + length * j);
	}
	double no_multithreading = omp_get_wtime() - now;

	now = omp_get_wtime();
#pragma omp parallel for 
	for (int j = 0; j < number; j++)
	{			
		sort(B + length * j);
	}
	double multithreading = omp_get_wtime() - now;
	double speedup = no_multithreading / multithreading;

	/*
	for (int i = 0; i < number * length; i++)
        {
                printf("%f\n", A[i]);
        }
	*/

	printf("Speedup for Bubble sort on %d arrays of length %d: %f\n", number, length, speedup);

	free(A); A = NULL;
	free(B); B = NULL;
	return 0;
}
