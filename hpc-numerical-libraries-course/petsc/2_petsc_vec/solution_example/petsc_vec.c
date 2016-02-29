/**
 * @file 1_petsc_vec.c
 * @author Simone Bnà
 * @date 19 Feb 2016
 * @brief File containing the basic example of petsc vec usage.
 * source petsc_load_env.sh
 * make
 * qsub petsc_qsub_script.sh
 */

static const char help[] = "Petsc vec example.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

  PetscMPIInt    rank;
  PetscInt       i,istart,iend,n = 6,size_local, size_global;
  PetscScalar    *avec;
  Vec            x;

  /* Initialize the Petsc environment */
  PetscInitialize(&argc,&argv,(char*)0,help);
  
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscOptionsGetInt(NULL,"-n",&n,NULL);

  /*
    Create a vector, specifying only its global dimension.
    When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
    the vector format (currently parallel or sequential) is
    determined at runtime.  Also, the parallel partitioning of
    the vector is determined by PETSc at runtime.
  */
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,n);
  VecSetFromOptions(x);

  /*
    PETSc parallel vectors are partitioned by
    contiguous chunks of rows across the processors.  Determine
    which vector are locally owned.
  */
  VecGetOwnershipRange(x,&istart,&iend);
  VecGetSize(x, &size_global);
  VecGetLocalSize(x, &size_local);

  /* 
    Set the vector elements.
      - Always specify global locations of vector entries.
      - Each processor can insert into any location, even ones it does not own
  */
  
  /* 
    In this case each processor adds values to all the entries,
    this is not practical, but is merely done as an example
  */
  /*for (i=0; i<n; i++) {
    v    = (PetscReal)(rank*i);
    VecSetValues(x,1,&i,&v,ADD_VALUES);
  }*/

  /* 
    In this case each processor adds values to only its entries.
  */
  /*for (i=istart; i<iend; i++) {
    v    = (PetscReal)(i);
    VecSetValues(x,1,&i,&v,ADD_VALUES);
  }*/

  VecGetArray(x,&avec);
  for (i = 0; i < size_local; i++) {
    avec[i] = (PetscReal)(i) + (PetscReal)(size_local*rank);
  }
  VecRestoreArray(x,&avec);
  
  /*
    Assemble vector, using the 2-step process:
      VecAssemblyBegin(), VecAssemblyEnd()
    Computations can be done while messages are in transition
    by placing code between these two statements.
  */
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

  /*VecView(x, PETSC_VIEWER_STDOUT_WORLD);*/

  /* Print local and global size of the x vector */
  PetscPrintf(PETSC_COMM_SELF, "[%d] global size: %d , local size: %d, my idx range: [%d-%d] \n", \
              rank, size_global, size_local, istart, iend);

  /* Free memory */
  VecDestroy(&x);
  PetscFinalize();

  PetscFunctionReturn(0);
}
