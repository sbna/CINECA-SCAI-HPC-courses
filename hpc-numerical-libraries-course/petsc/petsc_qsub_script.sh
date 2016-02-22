#!/bin/bash/
#PBS -l walltime=1:00:00
#PBS -l select=2:ncpus=10:mpiprocs=10    # 2 nodes, 10 procs/node = 20 MPI tasks
#PBS -o job.out
#PBS -e job.err
#PBS -A cineca_cin

PBS_O_WORKDIR = /galileo/home/sbna0000/
cd $PBS_O_WORKDIR ! this is the dir where the job was submitted from

module autoload petsc
mpirun ./4_petsc_ksp_poisson -da_grid_x 64 -da_grid_y 64
