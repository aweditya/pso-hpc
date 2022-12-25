#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>
#include "omp.h"

#define PI acos(-1.0)
#define DIM 1

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

/******* Logging and Tracing *************/
FILE *journal = NULL;

void trace_init()
{
    journal = fopen("pso_trace.txt", "w");
}

void trace_deinit()
{
    if (NULL != journal)
        fclose(journal);
}

void trace_write(char *cmd, ssize_t cmdlen)
{
    if (NULL != journal)
    {
        if (cmdlen <= 0)
            cmd = NULL;
        fprintf(journal, "PSO>%s\n", cmd);
    }
}

void trace_read(char *cmd, ssize_t cmdlen)
{
    if (NULL != journal)
    {
        if (cmdlen <= 0)
            cmd = NULL;
        fprintf(journal, "NGS>%s\n", cmd);
    }
}

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

int ngspice_transact(int write_fd, char *cmd, ssize_t cmdlen, int read_fd, char *readbuf, ssize_t buflen)
{
    if (write(write_fd, cmd, cmdlen) < 0)
    {
        perror("Unable to write to the child process\n");
    }

    usleep(1000);
    if (buflen > 0)
    {
        ssize_t nread = read(read_fd, readbuf, buflen);
        if (nread < 0)
        {
            perror("Unable to read from the child process\n");
        }
    }

    return 0;
}

int get_objective(char *buf, ssize_t buflen, double *objective)
{
    char *result = strstr(buf, " = ");
    int retval = -1;
    if (NULL != result)
    {
        result += strlen(" = ");
        char *end_of_value = strstr(result, "\n");
        if (NULL != end_of_value)
        {
            *end_of_value = '\0';
            char data[32] = {};
            strncpy(data, result, strlen(result));
            *objective = atof(data);
            retval = 0;
        }
        else
        {
            perror("Unexpected end of value\n");
        }
    }
    // printf("%s %lf\n", buf, objective);
    return retval;
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

void find_overall_best_fit(particle_t *particles, int num_particles, double *overall_best_fit, int *index_gbest, int read_fd, int write_fd)
{
    // #pragma omp parallel for
    for (int i = 0; i < num_particles; i++)
    {
        // Compute fitness
        double r2 = particles[i].position.coordinate[0];
        char r2_string[10];
        snprintf(r2_string, 10, "%lf", r2);

        char *alter_cmd = "alter r2 ";
        char *k_cmd = "k\n";
        char *cmd;
        cmd = malloc(strlen(alter_cmd) + 10 + 3);
        strcpy(cmd, alter_cmd);
        strcat(cmd, r2_string);
        strcat(cmd, k_cmd);

        ngspice_transact(write_fd, cmd, strlen(cmd), read_fd, NULL, 0);

        char readbuf[256];
        memset(readbuf, 0, 256);

        cmd = "op\n";
        ngspice_transact(write_fd, cmd, strlen(cmd), read_fd, readbuf, sizeof(readbuf));

        memset(readbuf, 0, 256);
        cmd = "print i(vdd)*v(2)\n";
        ngspice_transact(write_fd, cmd, strlen(cmd), read_fd, readbuf, sizeof(readbuf));

        // Compute fitness
        double current_fitness = 0;
        if (get_objective(readbuf, strlen(readbuf), &current_fitness) == 0)
        {
            particles[i].fitness = current_fitness;
        }

        // Find best fitness
        if (current_fitness < *overall_best_fit)
        {
            // #pragma omp critical
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

int pso_main(int num_particles, int n_pso, int print_freq, int print_stats, int read_fd, int write_fd)
{
    unsigned int seed = 1;
    srand(seed);

    FILE *fp_avg = NULL, *fp_snap = NULL;

    // Create an array of seeds, one for each thread
    unsigned int *seeds;
    seeds = malloc(num_particles * sizeof(unsigned int));
    for (int i = 0; i < num_particles; i++)
    {
        seeds[i] = rand();
    }

    particle_t *particles;
    particles = malloc(num_particles * sizeof(particle_t));

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

    // double now = omp_get_wtime();
    init_particles(particles, num_particles, seeds);

    for (int iter = 0; iter < n_pso; iter++)
    {
        if (print_stats)
        {
            dump_stats(particles, num_particles, iter, print_freq, fp_avg, fp_snap);
        }

        overall_best_fit = __DBL_MAX__;
        find_overall_best_fit(particles, num_particles, &overall_best_fit, &index_gbest, read_fd, write_fd);

        for (int j = 0; j < DIM; j++)
        {
            overall_best_position.coordinate[j] = particles[index_gbest].position.coordinate[j];
        }

        printf("p_gbest[0] = %11.4e\n", overall_best_position.coordinate[0]);

        process_particle(particles, num_particles, overall_best_position, seeds);
    }

    // printf("Execution time: %lf\n", omp_get_wtime() - now);

    if (print_stats)
        close_stats(fp_avg, fp_snap);

    free(particles);
    particles = NULL;

    return 0;
}

// minimise f(x,y) = sin x * cos y + 0.25*x using PSO.
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

    int pid;
    /* pipefd1: master -> child
     * pipefd1[0] is the read end
     * pipefd1[1] is the write end
     *
     * pipefd2: child -> master
     * pipefd2[0] is the read end
     * pipefd2[1] is the write end*/
    int pipefd1[2], pipefd2[2];
    if (pipe(pipefd1) < 0 || pipe(pipefd2) < 0)
    {
        /* FATAL: cannot create pipe */
        perror("Cannot create pipe\n");
    }

#define PARENT_WRITE pipefd1[1]
#define CHILD_READ pipefd1[0]
#define CHILD_WRITE pipefd2[1]
#define PARENT_READ pipefd2[0]

    if ((pid = fork()) < 0)
    {
        /* Cannot fork process */
    }
    else if (pid == 0)
    {
        /* Inside the child process */
        close(PARENT_WRITE);
        close(PARENT_READ);

        if (dup2(CHILD_READ, STDIN_FILENO) < 0)
        {
            perror("Cannot overload STDIN in child process\n");
        }

        if (dup2(CHILD_WRITE, STDOUT_FILENO) < 0)
        {
            perror("Cannot overload STDOUT in child process\n");
        }

        char *av[] = {"ngspice", "-p", NULL};
        if (execvp("ngspice", av) < 0)
        {
            perror("Cannot spawn NGSpice\n");
        }
    }
    else
    {
        /* Inside the master process */
        close(CHILD_WRITE);
        close(CHILD_READ);

        char init_msg[256];
        memset(init_msg, 0, 256);

        char *netlist_cmd = "test.cir\n";
        ngspice_transact(PARENT_WRITE, netlist_cmd, strlen(netlist_cmd), PARENT_READ, init_msg, sizeof(init_msg));

        pso_main(num_particles, n_pso, print_freq, print_stats, PARENT_READ, PARENT_WRITE);

        char *exit_cmd = "exit\n";
        ngspice_transact(PARENT_WRITE, exit_cmd, strlen(exit_cmd), PARENT_READ, NULL, 0);

        wait(NULL);
    }

    return 0;
}
