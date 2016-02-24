#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=2:mpiprocs=2
#PBS -o job.out
#PBS -e job.err
#PBS -A cin_staff

PBS_O_WORKDIR="/galileo/home/userinternal/sbna0000/software/CINECA-SCAI-HPC-courses/hpc-numerical-libraries-course/petsc"
cd ${PBS_O_WORKDIR} 

echo ${PBS_O_WORKDIR}

# modules to be loaded for petsc version 3.5.2
#module load intel/cs-xe-2015--binary
#module load intelmpi/5.0.2--binary
#module load petsc/3.5.2--intelmpi--5.0.2--binary

# modules to be loaded for petsc version 3.6.3
module load profile/advanced
module load intel/pe-xe-2016--binary
module load intelmpi/5.1.1--binary
module load petsc/3.6.3--intelmpi--5.1.1--binary

mpirun -np 2 ./4_petsc_ksp_poisson -da_grid_x 64 -da_grid_y 64 -ksp_monitor_true_residual -ksp_view