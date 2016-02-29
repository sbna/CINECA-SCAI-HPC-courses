/**
 * @file 1_petsc_hello.c
 * @author Simone Bnà
 * @date 19 Feb 2016
 * @brief File containing the basic example of petsc usage.
 * source petsc_load_env.sh
 * make
 * qsub petsc_qsub_script.sh
 */

static const char help[] = "Petsc Hello World.\n\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

   PetscErrorCode ierr;
   PetscMPIInt    rank, size;

   /*
     Every PETSc program should begin with the PetscInitialize() routine.
     argc, argv - These command line arguments are taken to extract the options
                  supplied to PETSc and options supplied to MPI.
     help       - When PETSc executable is invoked with the option -help,
                  it prints the various options that can be applied at
                  runtime.  The user can use the "help" variable place
                  additional help messages in this printout.
   */
   PetscInitialize(&argc,&argv,(char *)0,help);

   /*
     The following MPI calls return the number of processes
     being used and the rank of this process in the group.
   */
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_rank(PETSC_COMM_WORLD, &size);

   /*
     Here we would like to print only one message that represents
     all the processes in the group.  We use PetscPrintf() with the
     communicator PETSC_COMM_WORLD.  Thus, only one message is
     printed representing PETSC_COMM_WORLD, i.e., all the processors.
   */
   PetscPrintf(PETSC_COMM_WORLD,"Number of processors = %d, rank = %d\n",size,rank);

   /*
     Here a barrier is used to separate the two states.
   */
   MPI_Barrier(PETSC_COMM_WORLD);
   
   /*
     Here we simply use PetscPrintf() with the communicator PETSC_COMM_SELF
     (where each process is considered separately).  Thus, this time the
     output from different processes does not appear in any particular order.
   */
   ierr = PetscPrintf(PETSC_COMM_SELF,"Hello by proc %d!\n", rank);CHKERRQ(ierr);

   /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
     See the PetscFinalize() manpage for more information.
   */
   ierr = PetscFinalize(); CHKERRQ(ierr);

   PetscFunctionReturn(0);

}


