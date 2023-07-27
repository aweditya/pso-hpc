#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sharedspice.h"

#define PI acos(-1.0)
#define DIM 1

typedef struct _point
{
    double coordinate[DIM];
} point;

typedef struct _particle
{
    point position;
    double velocity[DIM];

    double fitness;

    point best_fit_position;
    double best_fit;
} particle;

double drand(const double low, const double high)
{
    return low + (high - low) * (double)rand() / (double)RAND_MAX;
}

/**
 * 0: send netlist
 * 1: alter command
 * 2: op command
 * 3: print command
 */
int state = 0;
double current_fitness;

int pso_main(int num_particles, int n_pso, int print_freq)
{
    unsigned int seed = 1;
    srand(seed);

    particle *particles;
    particles = (particle *)malloc(num_particles * sizeof(particle));

    point overall_best_position; // Coordinates of overall best
    double overall_best_fit;     // Overall best
    int index_gbest;             // Index of particle that found the overall best

    double sum[DIM], p_avg[DIM];
    FILE *fp, *fp1;

    double w = 0.4;  // Inertial weight
    double p1 = 1.0; // Cognitive coefficient
    double p2 = 1.0; // Sociological coefficient

    // Boundaries of parameter space
    double p_min[DIM], p_max[DIM];
    for (int i = 0; i < DIM; i++)
    {
        p_min[i] = 0.0;
        p_max[i] = 8.0;
    }

    // Randomly initialise particle positions and velocities
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].best_fit = __DBL_MAX__;
        for (int j = 0; j < DIM; j++)
        {
            particles[i].position.coordinate[j] = drand(p_min[j], p_max[j]);
            particles[i].velocity[j] = 0.0;
        }
    }

    fp = fopen("avg.dat", "w");
    fp1 = fopen("snap.dat", "w");
    for (int iter = 0; iter < n_pso; iter++)
    {
        // Write snapshot every n1 iterations
        if (iter % print_freq == 0)
        {
            for (int i = 0; i < num_particles; i++)
            {
                for (int j = 0; j < DIM; j++)
                {
                    fprintf(fp1, "%11.4e ", particles[i].position.coordinate[j]);
                }
                fprintf(fp1, "\n");
            }
            fprintf(fp1, "\n");
        }

        // Compute average position values (for output only)
        fprintf(fp, "%d ", iter);
        for (int i = 0; i < DIM; i++)
        {
            sum[i] = 0.0;
            for (int j = 0; j < num_particles; j++)
            {
                sum[i] += particles[j].position.coordinate[i];
            }
            p_avg[i] = sum[i] / (double)num_particles;
            fprintf(fp, "%11.4e ", p_avg[i]);
        }

        fprintf(fp, "\n");

        overall_best_fit = __DBL_MAX__;
        for (int i = 0; i < num_particles; i++)
        {
            // Compute fitness
            double r2 = particles[i].position.coordinate[0];
            char r2_string[10];
            snprintf(r2_string, 10, "%lf", r2);

            char alter_cmd[32] = "alter r2 ";
            char k_cmd[32] = "k\n";
            char cmd[32];

            strcpy(cmd, alter_cmd);
            strcat(cmd, r2_string);
            strcat(cmd, k_cmd);
            ngSpice_Command(cmd);
            state = 2;

            memset(cmd, 0, sizeof(cmd));
            strcpy(cmd, "op\n");
            ngSpice_Command(cmd);
            state = 3;

            memset(cmd, 0, sizeof(cmd));
            strcpy(cmd, "print i(vdd)*v(2)\n");
            ngSpice_Command(cmd);
            state = 1;

            // Find gbest
            particles[i].fitness = current_fitness;
            if (current_fitness < overall_best_fit)
            {
                overall_best_fit = current_fitness;
                index_gbest = i;
            }
            // printf("BEST: %lf\n", overall_best_position.coordinate[0]);
        }

        for (int j = 0; j < DIM; j++)
        {
            overall_best_position.coordinate[j] = particles[index_gbest].position.coordinate[j];
        }

        printf("p_gbest[0] = %11.4e\n", overall_best_position.coordinate[0]);

        for (int i = 0; i < num_particles; i++)
        {
            // Update pbest[i]
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
                double r1 = drand(0.0, 1.0), r2 = drand(0.0, 1.0);
                particles[i].velocity[j] = w * particles[i].velocity[j] + r1 * p1 * (particles[i].best_fit_position.coordinate[j] - particles[i].position.coordinate[j]) + r2 * p2 * (overall_best_position.coordinate[j] - particles[i].position.coordinate[j]);

                // Move the particles
                particles[i].position.coordinate[j] += particles[i].velocity[j];

                // If particles go outside parameter space, put them back in
                for (int j = 0; j < DIM; j++)
                {
                    if (particles[i].position.coordinate[j] < p_min[j])
                    {
                        particles[i].position.coordinate[j] = p_min[j];
                    }
                    if (particles[i].position.coordinate[j] > p_max[j])
                    {
                        particles[i].position.coordinate[j] = p_max[j];
                    }
                }
            }
        }

        // printf("%lf\n", overall_best_fit);
    }

    fclose(fp);
    fclose(fp1);

    free(particles);

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
    int n_pso = 20;     // Number of updates
    int print_freq = 4; // Print frequency

    if (argc == 4)
    {
        num_particles = atoi(argv[1]);
        n_pso = atoi(argv[2]);
        print_freq = atoi(argv[3]);
    }

    int ret = ngSpice_Init(ng_getchar, NULL, NULL, NULL, NULL, NULL, NULL);

    char netlist_cmd[32] = "mptt.cir\n";
    ngSpice_Command(netlist_cmd);
    state = 1;

    pso_main(num_particles, n_pso, print_freq);
}
