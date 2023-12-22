#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <mpi.h>
#include "omp.h"
#include "sharedspice.h"

#define ALTER_CMD_STATE 1
#define TRAN_CMD_STATE 2
#define PRINT_CMD_STATE 3

#define DIM 4
#define NUM_PARTICLES_PER_PROC 10

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

typedef struct _pso_hyper_params
{
    double p_min[DIM]; // Lower bound of particle space
    double p_max[DIM]; // Upper bound of particle space
    double w;          // Inertial weight
    double p1;         // Cognitive coefficient
    double p2;         // Sociological coefficient
} pso_hyper_params_t;

// Initialize the structure in one statement
pso_hyper_params_t g_pso_hyper_params = {
    .p_min = {1.0, 1.0, 1.0, 1.0},         // Initialize p_min array with values
    .p_max = {400.0, 400.0, 400.0, 400.0}, // Initialize p_max array with values
    .w = 0.4,                              // Initialize w
    .p1 = 1.0,                             // Initialize p1
    .p2 = 1.0                              // Initialize p2
};

particle_t particles[NUM_PARTICLES_PER_PROC];

/********** NGSpice callbacks ***********/
SendChar ng_getchar;
ControlledExit ng_exit;

bool will_unload = false;

typedef struct _c_ngspice_handler
{
    void *instanceHandle;
    int (*Init)(SendChar *, SendStat *, ControlledExit *, SendData *, SendInitData *, BGThreadRunning *, void *);
    int (*Init_Sync)(GetVSRCData *, GetISRCData *, GetSyncData *, int *, void *);
    int (*Command)(char *);
    int state;
    int thread_id;
} c_ngspice_handler_t;

c_ngspice_handler_t ngspice_instances[NUM_PARTICLES_PER_PROC];

/********** MPI stuff *************/
int world_size, world_rank;
const int root = 0;

typedef struct _proc_metadata
{
    double best_fit;
    double best_fit_position[DIM];
} proc_metadata_t;

void maxloc(void *in, void *inout, int *len, MPI_Datatype *dptr)
{
    proc_metadata_t *invals = (proc_metadata_t *)in;
    proc_metadata_t *inoutvals = (proc_metadata_t *)inout;

    int i, j;
    for (i = 0; i < *len; i++)
    {
        if (invals[i].best_fit > inoutvals[i].best_fit)
        {
            inoutvals[i].best_fit = invals[i].best_fit;
            for (j = 0; j < DIM; j++)
            {
                inoutvals[i].best_fit_position[j] = invals[i].best_fit_position[j];
            }
        }
    }

    return;
}

void create_MPI_proc_metadata(MPI_Datatype *proc_metadata_type, MPI_Op *maxloc_op)
{
    // Create MPI datatype for proc_metadata_t
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    int block_lengths[2] = {1, DIM};
    MPI_Aint displacements[2] = {offsetof(proc_metadata_t, best_fit), offsetof(proc_metadata_t, best_fit_position)};

    // Create struct datatype
    MPI_Type_create_struct(2, block_lengths, displacements, types, proc_metadata_type);
    MPI_Type_commit(proc_metadata_type);

    // Create MPI operation
    MPI_Op_create((MPI_User_function *)maxloc, true, maxloc_op);
}

void free_MPI_proc_metadata(MPI_Datatype *proc_metadata_type, MPI_Op *maxloc_op)
{
    MPI_Op_free(maxloc_op);
    MPI_Type_free(proc_metadata_type);
}

/********** Functions *************/
double drand(const double low, const double high, unsigned int *seed)
{
    return low + (high - low) * (double)rand_r(seed) / (double)RAND_MAX;
}

void init_particles(unsigned int *seeds)
{
// Randomly initialise particle positions and velocities
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES_PER_PROC; i++)
    {
        particles[i].best_fit = __DBL_MAX__;
        for (int j = 0; j < DIM; j++)
        {
            particles[i].position.coordinate[j] = drand(g_pso_hyper_params.p_min[j],
                                                        g_pso_hyper_params.p_max[j],
                                                        &seeds[i]);
            particles[i].velocity[j] = 0.0;
        }
    }
}

void find_overall_best_fit(double *overall_best_fit, double *overall_best_fit_position)
{
    /**
     * States of execution
     * 0: send netlist
     * 1: alter command
     * 2: tran command
     * 3: print command
     */
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES_PER_PROC; i++)
    {
        // Compute fitness
        char alter_cmd[32];

        double mn_w, mp_w;
        for (int j = 0; j < DIM; j++)
        {
            mn_w = particles[i].position.coordinate[j];
            mp_w = 2 * mn_w;

            snprintf(alter_cmd, 32, "alter @mn%d[w]=%lf", j + 2, mn_w);
            ngspice_instances[i].Command(alter_cmd);

            snprintf(alter_cmd, 32, "alter @mp%d[w]=%lf", j + 2, mp_w);
            ngspice_instances[i].Command(alter_cmd);
        }
        ngspice_instances[i].state = TRAN_CMD_STATE;

        char cmd[128] = "tran 10p 40n 0 10p\n";
        ngspice_instances[i].Command(cmd);
        ngspice_instances[i].state = PRINT_CMD_STATE;

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "meas tran fall_time trig v(in) val=0.5 rise=1 targ v(out5) val=0.5 fall=1\n");
        ngspice_instances[i].Command(cmd);

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "meas tran rise_time trig v(in) val=0.5 fall=1 targ v(out5) val=0.5 rise=1\n");
        ngspice_instances[i].Command(cmd);

        memset(cmd, 0, sizeof(cmd));
        strcpy(cmd, "print fall_time+rise_time\n");
        ngspice_instances[i].Command(cmd);
        ngspice_instances[i].state = ALTER_CMD_STATE;

#pragma omp critical
        // Find gbest
        if (particles[i].fitness < *overall_best_fit)
        {
            *overall_best_fit = particles[i].fitness;
            for (int j = 0; j < DIM; j++)
            {
                overall_best_fit_position[j] = particles[i].position.coordinate[j];
            }
        }
    }
}

void process_particle(point_t overall_best_position, unsigned int *seeds)
{
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES_PER_PROC; i++)
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
                                       r1 * g_pso_hyper_params.p1 * (particles[i].best_fit_position.coordinate[j] - particles[i].position.coordinate[j]) +
                                       r2 * g_pso_hyper_params.p2 * (overall_best_position.coordinate[j] - particles[i].position.coordinate[j]);

            // Move the particles
            particles[i].position.coordinate[j] += particles[i].velocity[j];

            // If particles go outside parameter space, put them back in
            for (int k = 0; k < DIM; k++)
            {
                if (particles[i].position.coordinate[j] < g_pso_hyper_params.p_min[k])
                {
                    particles[i].position.coordinate[j] = g_pso_hyper_params.p_min[k];
                }

                if (particles[i].position.coordinate[j] > g_pso_hyper_params.p_max[k])
                {
                    particles[i].position.coordinate[j] = g_pso_hyper_params.p_max[k];
                }
            }
        }
    }
}

int pso_main(int n_pso, int print_freq)
{
    srand(world_rank + 1); // Start seeds from 1, not 0

    // Create an array of seeds, one for each thread
    unsigned int seeds[NUM_PARTICLES_PER_PROC];
    for (int i = 0; i < NUM_PARTICLES_PER_PROC; i++)
    {
        seeds[i] = rand();
    }

    point_t overall_best_position;         // Coordinates of overall best
    proc_metadata_t proc_overall_best_fit; // Overall best for current process
    proc_metadata_t overall_best_fit;      // Overall best

    // create MPI datatype and operation
    MPI_Datatype proc_metadata_type;
    MPI_Op maxloc_op;
    create_MPI_proc_metadata(&proc_metadata_type, &maxloc_op);

    init_particles(seeds);
    for (int iter = 0; iter < n_pso; iter++)
    {
        proc_overall_best_fit.best_fit = __DBL_MAX__;
        find_overall_best_fit(&proc_overall_best_fit.best_fit, proc_overall_best_fit.best_fit_position);
        // Get overall best amongst all processes
        MPI_Allreduce(&proc_overall_best_fit, &overall_best_fit, 1, proc_metadata_type, maxloc_op, MPI_COMM_WORLD);

        for (int j = 0; j < DIM; j++)
        {
            overall_best_position.coordinate[j] = overall_best_fit.best_fit_position[j];
        }

        if (world_rank == root)
        {
            for (int i = 0; i < DIM; i++)
            {
                printf("p_gbest[%d] = %11.4e ", i, overall_best_position.coordinate[i]);
            }
            printf("\n");
        }

        process_particle(overall_best_position, seeds);
    }

    if (world_rank == 0)
    {
        printf("Overall Best Fit: %11.4e\n", overall_best_fit.best_fit);
    }

    free_MPI_proc_metadata(&proc_metadata_type, &maxloc_op);
    return 0;
}

void init_ngspice_instances(char *netlist_cmd)
{
    char *errmsg = NULL;
    char loadstring[32];
    for (int i = 0; i < NUM_PARTICLES_PER_PROC; i++)
    {
        sprintf(loadstring, "bin/libngspice%d.so", (i + 1) * (world_rank + 1));

        ngspice_instances[i].instanceHandle = dlopen(loadstring, RTLD_NOW);
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        if (!ngspice_instances[i].instanceHandle)
        {
            fprintf(stderr, "%s not loaded!\n", loadstring);
            exit(1);
        }

        ngspice_instances[i].Init = dlsym(ngspice_instances[i].instanceHandle, "ngSpice_Init");
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        ngspice_instances[i].Init_Sync = dlsym(ngspice_instances[i].instanceHandle, "ngSpice_Init_Sync");
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        ngspice_instances[i].Command = dlsym(ngspice_instances[i].instanceHandle, "ngSpice_Command");
        errmsg = dlerror();
        if (errmsg)
        {
            printf("%s\n", errmsg);
        }

        ngspice_instances[i].Init(ng_getchar, NULL, ng_exit, NULL, NULL, NULL, NULL);

        ngspice_instances[i].thread_id = i;
        ngspice_instances[i].Init_Sync(NULL, NULL, NULL, &ngspice_instances[i].thread_id, NULL);

        ngspice_instances[i].Command(netlist_cmd);
        ngspice_instances[i].state = ALTER_CMD_STATE;
    }
}

void close_ngspice_instances()
{
    for (int i = 0; i < NUM_PARTICLES_PER_PROC; i++)
    {
        dlclose(ngspice_instances[i].instanceHandle);
    }
}

int main(int argc, char **argv)
{
    char netlist_cmd[32] = "mos_buffer5y.cir\n";

    int n_pso = 20;     // Number of updates
    int print_freq = 4; // Print frequency

    if (argc == 4)
    {
        n_pso = atoi(argv[1]);
        print_freq = atoi(argv[2]);
    }

    MPI_Init(NULL, NULL);                       // initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // get rank of current process

    init_ngspice_instances(netlist_cmd);
    pso_main(n_pso, print_freq);
    close_ngspice_instances();

    MPI_Finalize();

    return 0;
}

int ng_getchar(char *outputreturn, int ident, void *userdata)
{
    if (ngspice_instances[ident].state == 3)
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
        dlclose(ngspice_instances[ident].instanceHandle);
    }
    else
    {
        printf("DNote: Prepare unloading ngspice%d\n", ident);
        will_unload = true;
    }

    return exitstatus;
}
