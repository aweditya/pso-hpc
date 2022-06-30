#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PI acos(-1.0)

double drand(const double low, const double high)
{
    return low + (high - low) * ((double)rand()) / ((double)RAND_MAX);
}

// minimise f(x,y) = sin x * cos y + 0.25*x using PSO.
int main(int argc, char **argv)
{
    int num_particles = 20;
    if (argc == 2)
    {
        num_particles = atoi(argv[1]);
    }

    double *p[num_particles]; // Coordinates of particles
    for (int i = 0; i < num_particles; i++)
    {
        p[i] = malloc(2 * sizeof(double));
    }

    double *p_pbest[num_particles]; // Coordinates of best value found by each particle
    for (int i = 0; i < num_particles; i++)
    {
        p_pbest[i] = malloc(2 * sizeof(double));
    }

    double *fvalue_pbest; // Best value found by each particle
    fvalue_pbest = malloc(num_particles * sizeof(double));

    double p_gbest[2]; // Coordinates of overall best
    double fvalue_gbest; // Overall best
    int index_gbest; // Index of particle that found the overall best

    double *v[num_particles]; // Velocities of particles
    for (int i = 0; i < num_particles; i++)
    {
        v[i] = malloc(2 * sizeof(double));
    }

    double *fvalue;
    fvalue = malloc(num_particles * sizeof(double));

    double sum[2], p_avg[2];
    FILE *fp, *fp1;

    double w = 0.4; // Inertial weight
    double p1 = 1.0; // Cognitive coefficient
    double p2 = 1.0; // Sociological coefficient

    // Boundaries of parameter space
    double p_min[2], p_max[2];
    p_min[0] = 0.0; p_max[0] = 8.0;
    p_min[1] = 0.0; p_max[1] = 8.0;

    int n_pso = 20; // Number of updates
    int n1 = 4;

    unsigned int iseed = 1;
    srand(iseed);

    // Randomly initialise particle positions and velocities
    for (int i = 0; i < num_particles; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            p[i][j] = drand(p_min[j], p_max[j]);
            v[i][j] = 0.0;
        }
    }

    fp = fopen("avg.dat", "w");
    fp1 = fopen("snap.dat", "w");
    for (int iter = 0; iter < n_pso; iter++)
    {
        // Write snapshot every n1 iterations
        if (iter % n1 == 0)
        {
            for (int i = 0; i < num_particles; i++)
            {
                fprintf(fp1, "%11.4e  %11.4e\n", p[i][0], p[i][1]);
            }
            fprintf(fp1, "\n");
        }

        // Compute average position values (for output only)
        sum[0] = 0.0;
        sum[1] = 0.0;
        for (int i = 0; i < num_particles; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                sum[j] += p[i][j];
            }
        }
        p_avg[0] = sum[0] / (double)num_particles;
        p_avg[1] = sum[1] / (double)num_particles;

        fprintf(fp, "%d %11.4e %11.4e\n", iter, p_avg[0], p_avg[1]);

        // Compute fitness
        for (int i = 0; i < num_particles; i++)
        {
            double x = p[i][0], y = p[i][1];

            // fvalue[i] = sin(x)*cos(y);
            fvalue[i] = sin(x) * cos(y) + 0.25 * x;
            if (iter == 0)
            {
                fvalue_pbest[i] = fvalue[i];
                p_pbest[i][0] = p[i][0];
                p_pbest[i][1] = p[i][1];
            }
        }

        // Find gbest
        fvalue_gbest = 100.0;
        for (int i = 0; i < num_particles; i++)
        {
            if (fvalue[i] < fvalue_gbest)
            {
                index_gbest = i;
                fvalue_gbest = fvalue[i];
            }
        }
        p_gbest[0] = p[index_gbest][0];
        p_gbest[1] = p[index_gbest][1];

        printf("p_gbest[0] = %11.4e p_gbest[1] = %11.4e \n", p_gbest[0], p_gbest[1]);

        // Update pbest[i]
        if (iter > 0)
        {
            for (int i = 0; i < num_particles; i++)
            {
                if (fvalue[i] < fvalue_pbest[i])
                {
                    fvalue_pbest[i] = fvalue[i];
                    p_pbest[i][0] = p[i][0];
                    p_pbest[i][1] = p[i][1];
                }
            }
        }

        // Update velocities
        for (int i = 0; i < num_particles; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                double r1 = drand(0.0, 1.0), r2 = drand(0.0, 1.0);
                v[i][j] = w * v[i][j] + r1 * p1 * (p_pbest[i][j] - p[i][j]) + r2 * p2 * (p_gbest[j] - p[i][j]);
            }
        }

        // Move the particles
        for (int i = 0; i < num_particles; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                p[i][j] = p[i][j] + v[i][j];
            }
        }

        // If particles go outside parameter space, put them back in
        for (int i = 0; i < num_particles; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if (p[i][j] < p_min[j])
                {
                    p[i][j] = p_min[j];
                }
                if (p[i][j] > p_max[j])
                {
                    p[i][j] = p_max[j];
                }
            }
        }
    }
    fclose(fp);
    fclose(fp1);
}