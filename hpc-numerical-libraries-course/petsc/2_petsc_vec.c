// file 2_petsc_vec.c
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
	
	PetscViewer viewer_fd;

	Vec va;

	int size_local, size_global, low_idx, high_idx;

	// Init stage

   PetscInitialize(&argc,&args,(char *)0,PETSC_NULL);

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	// Load vector

	PetscPrintf(PETSC_COMM_WORLD, "\n[%d] Loading data ...", rank);

  	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD , "data/va_200.bin" , FILE_MODE_READ , &viewer_fd ); CHKERRQ(ierr);
  	ierr = VecCreate(PETSC_COMM_WORLD, &va);
 	ierr = VecLoad(va, viewer_fd); CHKERRQ(ierr);
  	ierr = PetscViewerDestroy(&viewer_fd); CHKERRQ(ierr);

  	CHKMEMQ;

  	PetscPrintf(PETSC_COMM_WORLD, "\n[%d] Loading data done. \n", rank);

	// Vector operations

	VecView(va, PETSC_VIEWER_STDOUT_WORLD);

	VecGetSize(va, &size_global);
	VecGetLocalSize(va, &size_local);
	VecGetOwnershipRange(va, &low_idx, &high_idx);

	PetscPrintf(PETSC_COMM_SELF, "[%d] global size: %d , local size: %d, my idx range: [%d-%d] \n", rank, size_global, size_local, low_idx, high_idx-1);

	// Free memory
	
	VecDestroy(&va);
	ierr = PetscFinalize(); CHKERRQ(ierr);

   return 0;
}
