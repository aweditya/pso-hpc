#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "omp.h"

#define DIM 1000

/********** Type Definitions *************/
typedef struct _point
{
    double coordinate[DIM];
} point_t;

typedef struct _particle
{
    point_t position;
    double velocity[DIM];
    double fitness;
    point_t best_fit_position;
    double best_fit;
} particle_t;

/********** Global Variables *************/
double p_min[DIM]; // Lower bound of particle space
double p_max[DIM]; // Upper bound of particle space
double w = 0.8;    // Inertial weight
double p1 = 1.0;   // Cognitive coefficient
double p2 = 1.0;   // Sociological coefficient

/********** Functions  *************/
double drand(const double low, const double high, unsigned int *seed)
{
    return low + ((high - low) * ((double)rand_r(seed)) / ((double)RAND_MAX));
}

void init_problem(double *A[DIM], double *b)
{
    unsigned int seed = 1;
    for (int row = 0; row < DIM; row++)
    {
        b[row] = drand(-1.0, 1.0, &seed);
        for (int column = 0; column < DIM; column++)
        {
            A[row][column] = drand(-1.0, 1.0, &seed);
        }
    }
}

double compute_objective(double *A[DIM], double *x, double *b)
{
    double norm = 0.0;
    for (int i = 0; i < DIM; i++)
    {
        double product = 0.0;
        for (int j = 0; j < DIM; j++)
        {
            product += A[i][j] * x[j];
        }
        norm += (product - b[i]) * (product - b[i]);
    }
    return norm;
}

void init_particles(particle_t *particles, int num_particles)
{
    // Randomly initialise particle positions and velocities
#pragma omp parallel
    {
        unsigned int seed = 7391 + 17 * omp_get_thread_num();

// printf("seed = %d %d \n", seed, omp_get_thread_num());
#pragma omp for
        for (int i = 0; i < num_particles; i++)
        {
            particles[i].best_fit = __DBL_MAX__;
            for (int j = 0; j < DIM; j++)
            {
                particles[i].position.coordinate[j] = drand(p_min[j], p_max[j], &seed);
                // printf("Thread no. %d. coordinates[%d] %11.4e ", omp_get_thread_num(), j, particles[i].position.coordinate[j]);
                particles[i].velocity[j] = 0.0;
            }
            // printf("\n");
        }
    }
}

void init_stats(FILE **fp_avg, FILE **fp_snap)
{
    *fp_avg = fopen("avg.dat", "w");
    *fp_snap = fopen("snap.dat", "w");
}

void dump_stats(particle_t *particles, int num_particles, int iter, int print_freq, FILE *fp_avg, FILE *fp_snap)
{
    double sum[DIM], p_avg[DIM];

    // Write snapshot every print_freq iterations
    if (iter % print_freq == 0)
    {
        for (int i = 0; i < num_particles; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fprintf(fp_snap, "%11.4e ", particles[i].position.coordinate[j]);
            }
            fprintf(fp_snap, "\n");
        }
        fprintf(fp_snap, "\n");
    }

    // Compute average position values (for output only)
    for (int i = 0; i < DIM; i++)
    {
        sum[i] = 0.0;
    }

    for (int i = 0; i < num_particles; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            sum[j] += particles[i].position.coordinate[j];
        }
    }

    for (int i = 0; i < DIM; i++)
    {
        p_avg[i] = sum[i] / (double)num_particles;
    }

    fprintf(fp_avg, "%d ", iter);
    for (int i = 0; i < DIM; i++)
    {
        fprintf(fp_avg, "%11.4e ", p_avg[i]);
    }
    fprintf(fp_avg, "\n");
}

void close_stats(FILE *fp_avg, FILE *fp_snap)
{
    fclose(fp_avg);
    fclose(fp_snap);
}

void find_overall_best_fit(double *A[DIM], double *b, particle_t *particles, int num_particles, double *overall_best_fit, int *index_gbest)
{
#pragma omp parallel for
    for (int i = 0; i < num_particles; i++)
    {
        // Compute fitness
        particles[i].fitness = compute_objective(A, particles[i].position.coordinate, b);

        // Find best fitness
        double current_fitness = particles[i].fitness;
        if (current_fitness < *overall_best_fit)
        {
#pragma omp critical
            {
                *overall_best_fit = particles[*index_gbest].fitness;
                if (current_fitness < *overall_best_fit)
                {
                    *index_gbest = i;
                    *overall_best_fit = current_fitness;
                }
            }
        }
    }
}

void process_particle(particle_t *particles, int num_particles, point_t overall_best_position)
{
#pragma omp parallel
    {
        unsigned int seed = 7391 + 17 * omp_get_thread_num();

#pragma omp for
        for (int i = 0; i < num_particles; i++)
        {
            // Update best_fit of each particle
            if (particles[i].fitness < particles[i].best_fit)
            {
                particles[i].best_fit = particles[i].fitness;
                for (int j = 0; j < DIM; j++)
                {
                    particles[i].best_fit_position.coordinate[j] = particles[i].position.coordinate[j];
                }
            }

            for (int j = 0; j < DIM; j++)
            {
                // Update velocities
                double r1 = drand(0.0, 1.0, &seed);
                double r2 = drand(0.0, 1.0, &seed);
                particles[i].velocity[j] = w * particles[i].velocity[j] + r1 * p1 * (particles[i].best_fit_position.coordinate[j] - particles[i].position.coordinate[j]) +
                                           r2 * p2 * (overall_best_position.coordinate[j] - particles[i].position.coordinate[j]);

                // Move the particles
                particles[i].position.coordinate[j] += particles[i].velocity[j];

                // If particles go outside parameter space, put them back in
                for (int j = 0; j < DIM; j++)
                {
                    if (particles[i].position.coordinate[j] < p_min[j])
                        particles[i].position.coordinate[j] = p_min[j];

                    if (particles[i].position.coordinate[j] > p_max[j])
                        particles[i].position.coordinate[j] = p_max[j];
                }
            }
        }
    }
}

// minimise f(x) = \|Ax - b\|_{2}^{2}
int main(int argc, char **argv)
{
    int num_particles = 20;
    int n_pso = 500;     // Number of updates
    int print_stats = 0; // Dump stats or not
    int print_freq = 4;  // Print frequency
    FILE *fp_avg = NULL, *fp_snap = NULL;

    if (argc == 5)
    {
        num_particles = atoi(argv[1]);
        n_pso = atoi(argv[2]);
        print_stats = atoi(argv[3]);
        print_freq = atoi(argv[4]);
    }

    particle_t *particles;
    particles = malloc(num_particles * sizeof(particle_t));

    point_t overall_best_position; // Coordinates of overall best
    double overall_best_fit;       // Overall best
    int index_gbest = 0;           // Index of particle that found the overall best

    for (int i = 0; i < DIM; i++)
    {
        p_min[i] = -10.0;
        p_max[i] = 10.0;
    }

    if (print_stats)
        init_stats(&fp_avg, &fp_snap);

    double *A[DIM];
    for (int i = 0; i < DIM; i++)
    {
        A[i] = malloc(DIM * sizeof(double));
    }

    double *b;
    b = malloc(DIM * sizeof(double));
    init_problem(A, b);
    init_particles(particles, num_particles);

    double now = omp_get_wtime();
    for (int iter = 0; iter < n_pso; iter++)
    {
        if (print_stats)
        {
            dump_stats(particles, num_particles, iter, print_freq, fp_avg, fp_snap);
        }

        overall_best_fit = __DBL_MAX__;
        find_overall_best_fit(A, b, particles, num_particles, &overall_best_fit, &index_gbest);

        for (int j = 0; j < DIM; j++)
        {
            overall_best_position.coordinate[j] = particles[index_gbest].position.coordinate[j];
            // printf("p_gbest[%d] = %11.4e ", j, overall_best_position.coordinate[j]);
        }
        // printf("\n");

        process_particle(particles, num_particles, overall_best_position);
    }
    printf("Execution time: %lf\n", omp_get_wtime() - now);

    if (print_stats)
    {
        close_stats(fp_avg, fp_snap);
    }

    free(particles);
    particles = NULL;
    for (int i = 0; i < DIM; i++)
    {
        free(A[i]);
        A[i] = NULL;
    }
    free(b);
    b = NULL;
}