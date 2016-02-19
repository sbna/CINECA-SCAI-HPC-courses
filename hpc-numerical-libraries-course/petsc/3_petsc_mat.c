// file 3_petsc_mat.c
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

	Mat mC;

	int row_local_min, row_local_max, row_global, col_global;

	// Init stage

   PetscInitialize(&argc,&args,(char *)0,PETSC_NULL);

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	// Load matrix

	PetscPrintf(PETSC_COMM_WORLD, "\n[%d] Loading data ...", rank);

  	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD , "data/mC_200x200.bin" , FILE_MODE_READ , &viewer_fd ); CHKERRQ(ierr);
  	ierr = MatCreate(PETSC_COMM_WORLD, &mC);
  	ierr = MatSetType(mC, MATAIJ);
  	ierr = MatLoad(mC, viewer_fd); CHKERRQ(ierr);
  	ierr = PetscViewerDestroy(&viewer_fd); CHKERRQ(ierr);

  	CHKMEMQ;

  	PetscPrintf(PETSC_COMM_WORLD, "\n[%d] Loading data done. \n", rank);

	// Matrix operations

	//MatView(mC, PETSC_VIEWER_STDOUT_WORLD);

	MatGetSize(mC,&row_global,&col_global);
	MatGetOwnershipRange(mC,&row_local_min,&row_local_max);

	PetscPrintf(PETSC_COMM_SELF, "[%d] global size: %dx%d, range of matrix rows: %d-%d\n", 
                                 rank, row_global, col_global, row_local_min, row_local_max-1);

	// Free memory
	
	MatDestroy(&mC);
	ierr = PetscFinalize(); CHKERRQ(ierr);

   return 0;
}
