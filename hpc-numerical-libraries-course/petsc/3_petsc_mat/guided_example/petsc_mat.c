/**
 * @file 1_petsc_mat.c
 * @author Simone Bnà
 * @date 19 Feb 2016
 * @brief File containing the basic example of petsc mat usage.
 * source loadPetscEnv.sh 
 * make
 * qsub petscSubmissionScript
 */

static const char help[] = "Petsc mat example.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

   PetscErrorCode ierr;
   PetscMPIInt    rank;
   PetscMPIInt    nprocs;
   Mat            A;        
   PetscInt       i,j,Ii,J,Istart,Iend,m=8,n=8,global_rows,global_cols;
   PetscScalar    v;
   short unsigned int use_matrix_one_shot_creation = 1;

   /* Initialize the Petsc environment */
   PetscInitialize(&argc,&argv,(char*)0,help);

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
   
   /* Read from commandline the number of grid point in x and y direction */
   PetscOptionsGetInt(NULL,"-grid_x",&m,NULL);
   PetscOptionsGetInt(NULL,"-grid_y",&n,NULL);
 
   /*
       Create parallel matrix, specifying only its global dimensions.
       When using MatCreate(), the matrix format can be specified at
       runtime. Also, the parallel partitioning of the matrix is
       determined by PETSc at runtime. Matrix can be also created 
       in one shot using the MatCreateAIJ function.

       Performance tuning note:  For problems of substantial size,
       preallocation of matrix memory is crucial for attaining good
       performance. See the matrix chapter of the users manual for details.
    */
    if(!use_matrix_one_shot_creation) {
      MatCreate(PETSC_COMM_WORLD,&A);
      MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);
      MatSetFromOptions(A);
      if(nprocs > 1) {
        MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);
      }
      else {
        MatSeqAIJSetPreallocation(A,5,NULL);
      }
    }
    else {
      PetscPrintf(PETSC_COMM_WORLD, "Matrix one shot creation \n");
      MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,5,PETSC_NULL,5,PETSC_NULL,&A);
    }

    /*
      Currently, all PETSc parallel matrix formats are partitioned by
      contiguous chunks of rows across the processors.  Determine which
      rows of the matrix are locally owned.
    */
    MatGetOwnershipRange(A,&Istart,&Iend);

    /*
      Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
          locally (but any non-local elements will be sent to the
          appropriate processor during matrix assembly).
       - Always specify global rows and columns of matrix entries.
      Note: this uses the less common natural ordering that orders first
      all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
     instead of J = I +- m as you might expect. The more standard ordering
     would first do all variables for y = h, then y = 2h etc.

    */
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; i = Ii/n; j = Ii - i*n;
      if (i>0)   {J = Ii - n; MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);}
      if (i<m-1) {J = Ii + n; MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);}
      if (j>0)   {J = Ii - 1; MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);}
      if (j<n-1) {J = Ii + 1; MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);
    }

   /*
      Assemble matrix, using the 2-step process:
      MatAssemblyBegin(), MatAssemblyEnd()
      Computations can be done while messages are in transition
      by placing code between these two statements.
   */
   MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

   /* A is symmetric. Set symmetric flag. */
   MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);
	
   /* Get the global size of the matrix */
   MatGetSize(A,&global_rows,&global_cols);

   /* Print the global and local size of the matrix */
   PetscPrintf(PETSC_COMM_SELF, "rank [%d]: global size: %dx%d, range of matrix rows: %d-%d\n", 
                                 rank, global_rows, global_cols, Istart, Iend-1);

   /* Free memory */
   MatDestroy(&A);
   PetscFinalize();

   PetscFunctionReturn(0);
}
