#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include "omp.h"
#include "sharedspice.h"

#define PI acos(-1.0)
#define DIM 4
#define NUM_PARTICLES 20

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

typedef void *funptr_t;

/******** PSO hyperparameters **********/
typedef struct _pso_hyper_params
{
    double p_min[DIM]; // Lower bound of particle space
    double p_max[DIM]; // Upper bound of particle space
    double w;    // Inertial weight
    double p1;   // Cognitive coefficient
    double p2;   // Sociological coefficient
} pso_hyper_params_t ;

pso_hyper_params_t g_pso_hyper_params;

// Initialize the structure in one statement
pso_hyper_params_t g_pso_hyper_params = {
    .p_min = {0.0, 0.0}, // Initialize p_min array with values
    .p_max = {0.0, 0.0}, // Initialize p_max array with values
    .w = 0.4, // Initialize w
    .p1 = 1.0, // Initialize p1
    .p2 = 1.0  // Initialize p2
};

particle_t particles[NUM_PARTICLES];
int state[NUM_PARTICLES], thread_id[NUM_PARTICLES];
int *ret;

/********** NGSpice callbacks ***********/
SendChar ng_getchar;
ControlledExit ng_exit;

bool will_unload = false;
void *ngdllhandles[NUM_PARTICLES];

funptr_t ngSpice_Init_handles[NUM_PARTICLES], ngSpice_Init_Sync_handles[NUM_PARTICLES], ngSpice_Command_handles[NUM_PARTICLES];

/********** Functions  *************/
double drand(const double low, const double high, unsigned int *seed)
{
    return low + (high - low) * (double)rand_r(seed) / (double)RAND_MAX;
}

void init_particles(unsigned int *seeds)
{
// Randomly initialise particle positions and velocities
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].best_fit = __DBL_MAX__;
        for (int j = 0; j < DIM; j++)
        {
            particles[i].position.coordinate[j] = drand(g_pso_hyper_params.p_min[j], g_pso_hyper_params.p_max[j], &seeds[i]);
            particles[i].velocity[j] = 0.0;
        }
    }
}

void init_stats(FILE **fp_avg, FILE **fp_snap)
{
    *fp_avg = fopen("avg.dat", "w");
    *fp_snap = fopen("snap.dat", "w");
}

void dump_stats(int iter, int print_freq, FILE *fp_avg, FILE *fp_snap)
{
    double sum[DIM], p_avg[DIM];
    // Write snapshot every print_freq iterations
    if (iter % print_freq == 0)
    {
        for (int i = 0; i < NUM_PARTICLES; i++)
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
        for (int j = 0; j < NUM_PARTICLES; j++)
        {
            sum[i] += particles[j].position.coordinate[i];
        }
        p_avg[i] = sum[i] / NUM_PARTICLES;
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
void find_overall_best_fit(double *overall_best_fit, int *index_gbest)
{
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        // Compute fitness
        char alter_cmd[32];

        double mn_w, mp_w;
        for (int j = 0; j < DIM; j++)
        {
            mn_w = particles[i].position.coordinate[j];
            mp_w = 2 * mn_w;

            snprintf(alter_cmd, 32, "alter @mn%d[w]=%lf", j + 2, mn_w);
            // printf("%s\n", alter_cmd);
            ret = ((int *(*)(char *))ngSpice_Command_handles[i])(alter_cmd);

            snprintf(alter_cmd, 32, "alter @mp%d[w]=%lf", j + 2, mp_w);
            // printf("%s\n", alter_cmd);
            ret = ((int *(*)(char *))ngSpice_Command_handles[i])(alter_cmd);
        }
        state[i] = 2;

        char cmd[128] = "tran 10p 40n 0 10p\n";
        ret = ((int *(*)(char *))ngSpice_Command_handles[i])(cmd);
        state[i] = 3;

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "meas tran fall_time trig v(in) val=0.5 rise=1 targ v(out5) val=0.5 fall=1\n");
        ret = ((int *(*)(char *))ngSpice_Command_handles[i])(cmd);

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "meas tran rise_time trig v(in) val=0.5 fall=1 targ v(out5) val=0.5 rise=1\n");
        ret = ((int *(*)(char *))ngSpice_Command_handles[i])(cmd);

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "print fall_time+rise_time\n");
        ret = ((int *(*)(char *))ngSpice_Command_handles[i])(cmd);
        state[i] = 1;

#pragma omp critical
        // Find gbest
        if (particles[i].fitness < *overall_best_fit)
        {
            *overall_best_fit = particles[i].fitness;
            *index_gbest = i;
        }
    }
}

void process_particle(point_t overall_best_position, unsigned int *seeds)
{
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; i++)
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
            particles[i].velocity[j] = g_pso_hyper_params.w * particles[i].velocity[j] + 
                                    r1 * g_pso_hyper_params.p1 * (particles[i].best_fit_position.coordinate[j] - 
                                    particles[i].position.coordinate[j]) +
                                    r2 * g_pso_hyper_params.p2 * (overall_best_position.coordinate[j] - particles[i].position.coordinate[j]);

            // Move the particles
            particles[i].position.coordinate[j] += particles[i].velocity[j];

            // If particles go outside parameter space, put them back in
            for (int j = 0; j < DIM; j++)
            {
                if (particles[i].position.coordinate[j] < g_pso_hyper_params.p_min[j])
                    particles[i].position.coordinate[j] = g_pso_hyper_params.p_min[j];

                if (particles[i].position.coordinate[j] > g_pso_hyper_params.p_max[j])
                    particles[i].position.coordinate[j] = g_pso_hyper_params.p_max[j];
            }
        }
    }
}

int pso_main(int n_pso, int print_freq, int print_stats)
{
    unsigned int seed = 1;
    srand(seed);

    FILE *fp_avg = NULL, *fp_snap = NULL;

    // Create an array of seeds, one for each thread
    unsigned int seeds[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        seeds[i] = rand();
    }

    point_t overall_best_position; // Coordinates of overall best
    double overall_best_fit;       // Overall best
    int index_gbest = 0;           // Index of particle that found the overall best

    if (print_stats)
        init_stats(&fp_avg, &fp_snap);

    // double now = omp_get_wtime();
    init_particles(seeds);

    for (int iter = 0; iter < n_pso; iter++)
    {
        if (print_stats)
        {
            dump_stats(iter, print_freq, fp_avg, fp_snap);
        }

        overall_best_fit = __DBL_MAX__;
        find_overall_best_fit(&overall_best_fit, &index_gbest);

        for (int j = 0; j < DIM; j++)
        {
            overall_best_position.coordinate[j] = particles[index_gbest].position.coordinate[j];
        }

        for (int i = 0; i < DIM; i++)
        {
            printf("p_gbest[%d] = %11.4e ", i, overall_best_position.coordinate[i]);
        }
        printf("\n");

        process_particle(overall_best_position, seeds);
    }

    printf("Overall Best Fit: %11.4e\n", overall_best_fit);

    // printf("Execution time: %lf\n", omp_get_wtime() - now);

    if (print_stats)
        close_stats(fp_avg, fp_snap);

    return 0;
}

void load_ngspice(char *netlist_cmd)
{
    char *errmsg = NULL;
    char loadstring[32];
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        sprintf(loadstring, "bin/libngspice%d.so", i + 1);

        ngdllhandles[i] = dlopen(loadstring, RTLD_NOW);
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        if (!ngdllhandles[i])
        {
            fprintf(stderr, "%s not loaded!\n", loadstring);
            exit(1);
        }

        ngSpice_Init_handles[i] = dlsym(ngdllhandles[i], "ngSpice_Init");
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        ngSpice_Init_Sync_handles[i] = dlsym(ngdllhandles[i], "ngSpice_Init_Sync");
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        ngSpice_Command_handles[i] = dlsym(ngdllhandles[i], "ngSpice_Command");
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        ret = ((int *(*)(SendChar *, SendStat *, ControlledExit *, SendData *, SendInitData *,
                         BGThreadRunning *, void *))ngSpice_Init_handles[i])(ng_getchar,
                                                                             NULL, ng_exit, NULL, NULL, NULL, NULL);

        thread_id[i] = i;
        ret = ((int *(*)(GetVSRCData *, GetISRCData *, GetSyncData *, int *,
                         void *))ngSpice_Init_Sync_handles[i])(NULL, NULL, NULL, &thread_id[i], NULL);

        ret = ((int *(*)(char *))ngSpice_Command_handles[i])(netlist_cmd);
        state[i] = 1;
    }
}

int main(int argc, char **argv)
{
    char netlist_cmd[32] = "mos_buffer5y.cir\n";

    int n_pso = 20;      // Number of updates
    int print_stats = 0; // Dump stats or not
    int print_freq = 4;  // Print frequency

    if (argc == 4)
    {
        n_pso = atoi(argv[1]);
        print_stats = atoi(argv[2]);
        print_freq = atoi(argv[3]);
    }

    load_ngspice(netlist_cmd);

    pso_main(n_pso, print_freq, print_stats);

    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        dlclose(ngdllhandles[i]);
    }

    return 0;
}

int ng_getchar(char *outputreturn, int ident, void *userdata)
{
    if (state[ident] == 3)
    {
        char *result = strstr(outputreturn, " = ");
        if (result != NULL)
        {
            result += strlen(" = ");
            particles[ident].fitness = atof(result);
        }
        else
        {
            particles[ident].fitness = __DBL_MAX__;
        }
    }

    return 0;
}

int ng_exit(int exitstatus, bool immediate, bool quitexit, int ident, void *userdata)
{
    if (quitexit)
    {
        printf("DNote: Returned quit from library %d with exit status %d\n", ident, exitstatus);
    }
    if (immediate)
    {
        printf("DNote: Unload ngspice%d\n", ident);
        dlclose(ngdllhandles[ident]);
    }
    else
    {
        printf("DNote: Prepare unloading ngspice%d\n", ident);
        will_unload = true;
    }

    return exitstatus;
}
