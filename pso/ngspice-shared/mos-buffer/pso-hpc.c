#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "omp.h"
#include "sharedspice.h"

#define PI acos(-1.0)
#define DIM 4

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
            for (int j = 0; j < DIM; j++)
            {
                fprintf(fp_snap, "%11.4e ", particles[i].position.coordinate[j]);
            }
            fprintf(fp_snap, "\n");
        }
        fprintf(fp_snap, "\n");
    }

    // Compute average position values (for output only)
    fprintf(fp_avg, "%d ", iter);
    for (int i = 0; i < DIM; i++)
    {
        sum[i] = 0.0;
        for (int j = 0; j < num_particles; j++)
        {
            sum[i] += particles[j].position.coordinate[i];
        }
        p_avg[i] = sum[i] / (double)num_particles;
        fprintf(fp_avg, "%11.4e ", p_avg[i]);
    }
    fprintf(fp_avg, "\n");
}

void close_stats(FILE *fp_avg, FILE *fp_snap)
{
    fclose(fp_avg);
    fclose(fp_snap);
}

/**
 * 0: send netlist
 * 1: alter command
 * 2: tran command
 * 3: print command
*/
int state = 0;
double current_fitness;

void find_overall_best_fit(particle_t *particles, int num_particles, double *overall_best_fit, int *index_gbest)
{
    for (int i = 0; i < num_particles; i++)
    {
        // Compute fitness
        char reset_cmd[8] = "reset\n";
        ngSpice_Command(reset_cmd);

        char index_string[10];
        char alter_cmd[32];
        char mn_w_string[10];
        char mp_w_string[10];

        double mn_w, mp_w;
        for (int j = 0; j < DIM; j++)
        {
            snprintf(index_string, 2, "%d", (j + 2));

            mn_w = particles[i].position.coordinate[j];
            mp_w = 2 * mn_w;
            snprintf(mn_w_string, 10, "%lf", mn_w);
            snprintf(mp_w_string, 10, "%lf", mp_w);

            memset(alter_cmd, 0, sizeof(alter_cmd));
            strcpy(alter_cmd, "alter @mn");
            strcat(alter_cmd, index_string);
            strcat(alter_cmd, "[w]= ");
            strcat(alter_cmd, mn_w_string);
            // printf("%s\n", alter_cmd);
            ngSpice_Command(alter_cmd);

            memset(alter_cmd, 0, sizeof(alter_cmd));
            strcpy(alter_cmd, "alter @mp");
            strcat(alter_cmd, index_string);
            strcat(alter_cmd, "[w]= ");
            strcat(alter_cmd, mp_w_string);
            // printf("%s\n", alter_cmd);
            ngSpice_Command(alter_cmd);
        }
        state = 2;

        char cmd[128] = "tran 10p 40n 0 10p\n";
        ngSpice_Command(cmd);
        state = 3;

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "meas tran energy integ i(vdd) from=0.5n to=30.5n\n");
        ngSpice_Command(cmd);

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "print energy\n");
        ngSpice_Command(cmd);
        state = 1;

        // Find gbest
        current_fitness *= -1;
        // printf("%11.4e\n", current_fitness);
        particles[i].fitness = current_fitness;
        if (current_fitness < *overall_best_fit)
        {
            *overall_best_fit = current_fitness;
            *index_gbest = i;
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

int pso_main(int num_particles, int n_pso, int print_freq, int print_stats)
{
    unsigned int seed = 1;
    srand(seed);

    FILE *fp_avg = NULL, *fp_snap = NULL;

    // Create an array of seeds, one for each thread
    unsigned int *seeds;
    seeds = new unsigned int[num_particles];
    for (int i = 0; i < num_particles; i++)
    {
        seeds[i] = rand();
    }

    particle_t *particles;
    particles = new particle_t[num_particles];

    point_t overall_best_position; // Coordinates of overall best
    double overall_best_fit;       // Overall best
    int index_gbest = 0;           // Index of particle that found the overall best

    for (int i = 0; i < DIM; i++)
    {
        p_min[i] = 1.0;
        p_max[i] = 400.0;
    }

    if (print_stats)
        init_stats(&fp_avg, &fp_snap);

    // double now = omp_get_wtime();
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

        for (int i = 0; i < DIM; i++)
        {
            printf("p_gbest[%d] = %11.4e ", i, overall_best_position.coordinate[i]);
        }
        printf("\n");

        process_particle(particles, num_particles, overall_best_position, seeds);
    }

    printf("Overall Best Fit: %11.4e\n", overall_best_fit);

    // printf("Execution time: %lf\n", omp_get_wtime() - now);

    if (print_stats)
        close_stats(fp_avg, fp_snap);

    delete [] particles;
    delete [] seeds;

    return 0;
}

int ng_getchar(char *outputreturn, int ident, void *userdata)
{
    if (state == 3)
    {
        char *result = strstr(outputreturn, " = ");
        result += strlen(" = ");
        current_fitness = atof(result);
    }

    return 0;
}

int main(int argc, char **argv)
{
    int num_particles = 20;
    int n_pso = 20;      // Number of updates
    int print_stats = 0; // Dump stats or not
    int print_freq = 4;  // Print frequency

    if (argc == 5)
    {
        num_particles = atoi(argv[1]);
        n_pso = atoi(argv[2]);
        print_stats = atoi(argv[3]);
        print_freq = atoi(argv[4]);
    }

    ngSpice_Init(ng_getchar, NULL, NULL, NULL, NULL, NULL, NULL);

    char netlist_cmd[32] = "mos_buffer5y.cir\n";
    ngSpice_Command(netlist_cmd);
    state = 1;

    pso_main(num_particles, n_pso, print_freq, print_stats);

    return 0;
}
