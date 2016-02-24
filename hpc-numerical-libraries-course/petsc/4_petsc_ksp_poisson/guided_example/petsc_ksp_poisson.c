static const char help[] = "Solution of -Laplacian u = b using KSP and DMDA.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>  

PetscErrorCode ComputeManufacturedSolution(DM dm, Vec* sol); 
PetscErrorCode AssemblyRhs(DM dm, Vec* b); 
PetscErrorCode AssemblyMatrix(DM dm, Mat* A); 

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  /* 
    Initialize program
  */
  PetscInitialize(&argc,&argv,(char*)0,help);

  /* 
    Write a PETSc program that solve in parallel the following PDE:
      
        -nabla^2 u = -32*(x*(x-1) + y*(y-1)) in a box domain [0,1][0,1]
    
    with the following boundary conditions
      
        u = 0 in x=0, x=1, y=0 and y=1    

    The solution is:
      
        u = 16 * x * (x - 1) * y * (y - 1) 

    Use a distributed array (DMDA) to manage the parallel structured mesh 
    (with uniform coordinates) and vectors.
    Use the AssemblyMatrix and Assembly Rhs functions to create 
    the linear operator A and the rhs b of the linear system A x = b.
    Solve the problem with a KSP object (default one is 
    GMRES preconditioned by ILU (Block-Jacobi in parallel)).
    Check if the numerical solution is close in l2norm to the 
    the analytical solution (Use the ComputeManufacturedSolution).
  */


  /*   
    Finalize the program.
  */  
  PetscFinalize();
  
  PetscFunctionReturn(0);

}


/* 
  Assembly the Matrix corresponding to the finite
  difference discretization of the Laplacian operator. 
*/
PetscErrorCode AssemblyMatrix(DM dm, Mat* A)
{
  PetscInt       i,j,nrows = 0;
  MatStencil     col[5],row,*rows;
  PetscScalar    v[5],hx,hy,hxdhy,hydhx;
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  
  ierr  = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);

  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = PetscMalloc1(info.ym*info.xm,&rows);CHKERRQ(ierr);
  /*
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      row.j = j; row.i = i;
      /* boundary points */
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        v[0]            = 2.0*(hydhx + hxdhy);
        ierr            = MatSetValuesStencil(*A,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        rows[nrows].i   = i;
        rows[nrows++].j = j;
      } else {
        /* interior grid points */
        v[0] = -hxdhy;                                           col[0].j = j - 1; col[0].i = i;
        v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
        v[2] = 2.0*(hydhx + hxdhy);                              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
        v[4] = -hxdhy;                                           col[4].j = j + 1; col[4].i = i;
        ierr = MatSetValuesStencil(*A,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatZeroRowsColumnsStencil(*A,nrows,rows,2.0*(hydhx + hxdhy),NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(*A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  return 0;
}


/* 
  Assembly the Rhs with Dirichlet bdc.
*/
PetscErrorCode AssemblyRhs(DM dm, Vec* b) 
{
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  PetscScalar hx,hy;
  PetscInt i,j;  
  
  ierr  = DMDAGetLocalInfo(dm,&info); CHKERRQ(ierr);

  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);

  /* bdc for b */
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {     
      /* boundary points */
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        VecSetValue(*b, j+info.my*i, 0., INSERT_VALUES);
      }
      else {
        PetscScalar f = -32.*( hx*i*(hx*i - 1.) + hy*j*(hy*j - 1.) )*hx*hy;
        VecSetValue(*b, j+info.my*i, f, INSERT_VALUES);
      }
    }
  }

  VecAssemblyBegin(*b);
  VecAssemblyEnd(*b);

  return 0;
}


/* 
  Compute the manufactured solution.
*/
PetscErrorCode ComputeManufacturedSolution(DM dm, Vec* sol) 
{
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  PetscScalar hx,hy;
  PetscInt i,j;  

  ierr  = DMDAGetLocalInfo(dm,&info); CHKERRQ(ierr);

  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);

  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {     
        PetscScalar f = 16.*(hx*i)*(hx*i - 1.)*(hy*j)*(hy*j - 1.);
        VecSetValue(*sol, j+info.my*i, f, INSERT_VALUES);
    }
  }

  VecAssemblyBegin(*sol);
  VecAssemblyEnd(*sol);

  return 0;
}
