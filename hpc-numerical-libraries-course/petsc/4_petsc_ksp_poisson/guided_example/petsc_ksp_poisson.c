static const char help[] = "Solution of -Laplacian u = b using KSP and DMDA.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>  

PetscErrorCode ComputeManufacturedSolution(DM dm, Vec* sol); 
PetscErrorCode RhsAssembly(DM dm, Vec* b); 
PetscErrorCode FDLaplaceOperatorDiscretize(DM dm, Mat* A); 
double f(const double x, const double y);
double u(const double x, const double y);

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
  Evaluate the f function
  f= -32*(x*(x-1) + y*(y-1))
*/
#undef __FUNCT__
#define __FUNCT__ "f"
double f(const double x, const double y) {
  double f_x = -32.*(x*(x-1.) + y*(y-1.));
  return f_x;
}


/* 
  Evaluate the sol function
  sol u = 16 * x * (x - 1) * y * (y - 1) 
*/
#undef __FUNCT__
#define __FUNCT__ "u"
double u(const double x, const double y) {
  double sol =  16.*x*(x-1.)*y*(y-1.);
  return sol;
}


/* 
  Assemble the Matrix corresponding to the finite
  difference discretization of the Laplacian operator. 
*/
#undef __FUNCT__
#define __FUNCT__ "FDLaplaceOperatorDiscretize"
PetscErrorCode FDLaplaceOperatorDiscretize(DM dm, Mat* A)
{
  PetscInt       i,j,nrows = 0;
  MatStencil     col[5],row,*rows;
  PetscScalar    v[5],hx,hy,hxdhy,hydhx;
  DMDALocalInfo  info;
  
  DMDAGetLocalInfo(dm,&info);

  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);
  hxdhy = hx/hy;
  hydhx = hy/hx;

  PetscMalloc1(info.ym*info.xm,&rows);
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
        MatSetValuesStencil(*A,1,&row,1,&row,v,INSERT_VALUES);
        rows[nrows].i   = i;
        rows[nrows++].j = j;
      } else {
        /* interior grid points */
        v[0] = -hxdhy;                                           col[0].j = j - 1; col[0].i = i;
        v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
        v[2] = 2.0*(hydhx + hxdhy);                              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
        v[4] = -hxdhy;                                           col[4].j = j + 1; col[4].i = i;
        MatSetValuesStencil(*A,1,&row,5,col,v,INSERT_VALUES);
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);
  MatZeroRowsColumnsStencil(*A,nrows,rows,2.0*(hydhx + hxdhy),NULL,NULL);
  PetscFree(rows);

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  MatSetOption(*A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);

  PetscFunctionReturn(0);
}


/* 
  Assembly the Rhs with Dirichlet bdc.
*/
#undef __FUNCT__
#define __FUNCT__ "RhsAssembly"
PetscErrorCode RhsAssembly(DM dm, Vec* b) 
{
  DMDALocalInfo  info;
  PetscScalar hx,hy;
  PetscInt i,j;  
  PetscScalar f_x;
  double x, y;

  DMDAGetLocalInfo(dm,&info);

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
        x = (double)(hx*i);
        y = (double)(hy*j);
        f_x = f(x,y)*hx*hy;
        VecSetValue(*b, j+info.my*i, f_x, INSERT_VALUES);
      }
    }
  }

  VecAssemblyBegin(*b);
  VecAssemblyEnd(*b);

  PetscFunctionReturn(0);
}


/* 
  Compute the manufactured solution.
*/
#undef __FUNCT__
#define __FUNCT__ "ComputeManufacturedSolution"
PetscErrorCode ComputeManufacturedSolution(DM dm, Vec* sol) 
{
  DMDALocalInfo  info;
  PetscScalar hx,hy;
  PetscInt i,j;
  double x, y;
  PetscScalar val; 

  DMDAGetLocalInfo(dm,&info);

  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);

  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) { 
        x = (double)(hx*i);
        y = (double)(hy*j);    
        val = u(x, y);
        VecSetValue(*sol, j+info.my*i, val, INSERT_VALUES);
    }
  }

  VecAssemblyBegin(*sol);
  VecAssemblyEnd(*sol);

  PetscFunctionReturn(0);
}
