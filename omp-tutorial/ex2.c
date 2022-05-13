#include <stdio.h>
#include "omp.h"

#define NUM_THREADS 50

static long num_steps = 100000;
double step;

void main()
{
  double pi, sum=0.0;
  step = 1.0/(double) num_steps;

  double thread_sum[NUM_THREADS];

  int j;
  for (j = 0; j < NUM_THREADS; j++) {
    thread_sum[j] = 0.0;
  }

  double now = omp_get_wtime();
  
#pragma omp parallel num_threads(NUM_THREADS)
  {
    int i;
    double x;
    int thread_ID = omp_get_thread_num();
    
    int start = (num_steps / NUM_THREADS) * thread_ID;
    int stop = (num_steps / NUM_THREADS) * (thread_ID + 1); 
    
    for (i = start; i < stop; i++) {
      x = (i + 0.5)*step;
      thread_sum[thread_ID] = thread_sum[thread_ID] + 4.0/(1 + x*x);
    }   
  }

  for (j = 0; j < NUM_THREADS; j++) {
    sum = sum + thread_sum[j];
  }
  
  pi = step * sum;
  printf("Threads: %d \n", NUM_THREADS);
  printf("time: %f \n", omp_get_wtime() - now);
  printf("PI: %f \n", pi);
}
