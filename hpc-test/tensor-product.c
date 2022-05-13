#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define dimA 5000 
#define dimB 200
#define dimC 5000*200

double drand(double low, double high)
{
  return ( (double)rand() * ( high - low ) ) / (double)RAND_MAX + low;
}

int main()
{
  int i;
  int A[dimA], B[dimB], C[dimC], D[dimC];

  for (i = 0; i < dimA; i++) {
    A[i] = drand(0.0, 1.0);
  }
  for (i = 0; i < dimB; i++) {
    B[i] = drand(0.0, 1.0);
  }

  // No multithreading
  int p, q;
  double now = omp_get_wtime();
  for (p = 0; p < dimA; p++) {
    for (q = 0; q < dimB; q++) {
      C[dimB * p + q] = A[p] * B[q];
    }   
  }
  printf("Computation time without multithreading: %f\n", omp_get_wtime() - now);

  // Multithreading
  now = omp_get_wtime();
#pragma omp parallel
  {
    int j, k;
    #pragma omp for
      for (j = 0; j < dimA; j++) {
        for (k = 0; k < dimB; k++) {
          D[dimB * j + k] = A[j] * B[k];
        }
      }   
  }
  printf("Computation time with multithreading: %f\n", omp_get_wtime() - now);

}
