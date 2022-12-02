#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>
#include <math.h>

#define PI acos(-1.0)
#define DIM 1

struct point
{
    double coordinate[DIM];
};

typedef struct point point;

struct particle
{
    point position;
    double velocity[DIM];

    double fitness;

    point best_fit_position;
    double best_fit;
};

typedef struct particle particle;

double drand(const double low, const double high)
{
    return low + (high - low) * (double)rand() / (double)RAND_MAX;
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
        if (read(read_fd, readbuf, buflen) < 0)
        {
            perror("Unable to read from the child process\n");
        }
    }

    return 0;
}

double get_objective(char *buf, ssize_t buflen)
{
    int last_whitespace = 0;
    for (int i = 0; i < buflen; i++)
    {
        if (buf[i] == ' ')
        {
            last_whitespace = i;
        }
    }

    char* data;
    data = malloc(12);
    strncpy(data, buf + (last_whitespace + 1), 12);

    double objective = atof(data);
    // printf("%s %lf\n", buf, objective);
    return objective;   
}

int pso_main(int num_particles, int n_pso, int print_freq, int read_fd, int write_fd)
{
    unsigned int seed = 1;
    srand(seed);

    particle *particles;
    particles = malloc(num_particles * sizeof(particle));

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

            particles[i].fitness = get_objective(readbuf, strlen(readbuf));

            // Find gbest
            double current_fitness = particles[i].fitness;
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
    particles = NULL;

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

        pso_main(num_particles, n_pso, print_freq, PARENT_READ, PARENT_WRITE);

        char *exit_cmd = "exit\n";
        ngspice_transact(PARENT_WRITE, exit_cmd, strlen(exit_cmd), PARENT_READ, NULL, 0);

        wait(NULL);
    }
}
