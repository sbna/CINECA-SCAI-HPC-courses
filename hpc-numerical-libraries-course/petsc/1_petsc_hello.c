// file 1_petsc_hello.c
// source loadPetscEnv.sh 
// make
// qsub petscSubmissionScript

#include "petscksp.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{

   PetscErrorCode ierr;
   PetscMPIInt    rank;

   PetscInitialize(&argc,&args,(char *)0,PETSC_NULL);

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   ierr = PetscPrintf(PETSC_COMM_SELF,"Hello by procs %d!\n", rank);CHKERRQ(ierr);

   ierr = PetscFinalize(); CHKERRQ(ierr);

   return 0;
}
