#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "omp.h"

#define PI acos(-1.0)
#define DIM 2

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
double w = 0.4;    // Inertial weight
double p1 = 1.0;   // Cognitive coefficient
double p2 = 1.0;   // Sociological coefficient

/********** Functions  *************/
double drand(const double low, const double high, unsigned int *seed)
{
    return low + (high - low) * (double)rand_r(seed) / (double)RAND_MAX;
}

void init_particles(particle_t *particles, int num_particles, unsigned int *seeds)
{
// Randomly initialise particle positions and velocities
#pragma omp parallel for
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].best_fit = __DBL_MAX__;
        for (int j = 0; j < DIM; j++)
        {
            particles[i].position.coordinate[j] = drand(p_min[j], p_max[j], &seeds[i]);
            particles[i].velocity[j] = 0.0;
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
            fprintf(fp_snap, "%11.4e  %11.4e\n",
                    particles[i].position.coordinate[0], particles[i].position.coordinate[1]);
        }
        fprintf(fp_snap, "\n");
    }
    // Compute average position values (for output only)
    sum[0] = 0.0;
    sum[1] = 0.0;
    for (int i = 0; i < num_particles; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            sum[j] += particles[i].position.coordinate[j];
        }
    }
    p_avg[0] = sum[0] / (double)num_particles;
    p_avg[1] = sum[1] / (double)num_particles;

    fprintf(fp_avg, "%d %11.4e %11.4e\n", iter, p_avg[0], p_avg[1]);
}

void close_stats(FILE *fp_avg, FILE *fp_snap)
{
    fclose(fp_avg);
    fclose(fp_snap);
}

void find_overall_best_fit(particle_t *particles, int num_particles, double *overall_best_fit, int *index_gbest)
{
#pragma omp parallel for
    for (int i = 0; i < num_particles; i++)
    {
        // Compute fitness
        double x = particles[i].position.coordinate[0], y = particles[i].position.coordinate[1];
        particles[i].fitness = sin(x) * cos(y) + 0.25 * x;

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

void process_particle(particle_t *particles, int num_particles, point_t overall_best_position, unsigned int *seeds)
{
#pragma omp parallel for
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
            double r1 = drand(0.0, 1.0, &seeds[i]);
            double r2 = drand(0.0, 1.0, &seeds[i]);
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

// minimise f(x,y) = sin x * cos y + 0.25*x using PSO.
int main(int argc, char **argv)
{
    int num_particles = 20;
    int n_pso = 20;      // Number of updates
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

    unsigned int seed = 1;
    srand(seed);

    // Create an array of seeds, one for each thread
    unsigned int *seeds;
    seeds = (unsigned int *)malloc(num_particles * sizeof(unsigned int));
    for (int i = 0; i < num_particles; i++)
    {
        seeds[i] = rand();
    }

    particle_t *particles;
    particles = (particle_t *)malloc(num_particles * sizeof(particle_t));

    point_t overall_best_position; // Coordinates of overall best
    double overall_best_fit;       // Overall best
    int index_gbest = 0;           // Index of particle that found the overall best

    for (int i = 0; i < DIM; i++)
    {
        p_min[i] = 0.0;
        p_max[i] = 8.0;
    }

    if (print_stats)
        init_stats(&fp_avg, &fp_snap);

    double now = omp_get_wtime();
    init_particles(particles, num_particles, seeds);

    for (int iter = 0; iter < n_pso; iter++)
    {
        if (print_stats)
        {
            dump_stats(particles, num_particles, iter, print_freq, fp_avg, fp_snap);
        }

        overall_best_fit = __DBL_MAX__;
        find_overall_best_fit(particles, num_particles, &overall_best_fit, &index_gbest);

        for (int j = 0; j < DIM; j++)
        {
            overall_best_position.coordinate[j] = particles[index_gbest].position.coordinate[j];
        }

        printf("p_gbest[0] = %11.4e p_gbest[1] = %11.4e \n",
               overall_best_position.coordinate[0], overall_best_position.coordinate[1]);

        process_particle(particles, num_particles, overall_best_position, seeds);
    }

    printf("Execution time: %lf\n", omp_get_wtime() - now);

    if (print_stats)
        close_stats(fp_avg, fp_snap);

    free(particles);
    particles = NULL;
}