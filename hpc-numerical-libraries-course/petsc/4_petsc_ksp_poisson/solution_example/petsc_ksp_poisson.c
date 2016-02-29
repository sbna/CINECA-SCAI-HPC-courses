/**
 * @file 1_petsc_hello.c
 * @author Simone Bnà
 * @date 19 Feb 2016
 * @brief File containing the example of the solution in parallel 
 * of the following Poisson problem using KSP and DMDA:
 *
 *     -nabla^2 u = -32*(x*(x-1) + y*(y-1)) in a box domain [0,1][0,1]
 *   
 *   with the following boundary conditions
 *     
 *      u = 0 in x=0, x=1, y=0 and y=1    
 *
 *   The solution is:
 *     
 *       u = 16 * x * (x - 1) * y * (y - 1) 
 *
 * source petsc_load_env.sh
 * make
 * qsub petsc_qsub_script.sh
 */

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
  Vec            x,b,sol;         /* solution vectors */
  Mat            A;               /* linear operator */
  KSP            solver;
  DM             da;              /* DMDA object containing the structured mesh */
  PetscViewer    viewer; 
  PetscReal norm_sol, norm_diff;
  PetscErrorCode ierr;

  /* 
    Initialize program
  */

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* 
     -----------------------------------------------
       Mesh, matrix and rhs creation
     ----------------------------------------------- 
  */

  /* 
    Create a distributed array (DMDA) to manage the parallel structured mesh and vectors.
    The mesh is a 2D regular grid in a box domain [0,1][0,1]. 
    The default case is 4 nodes in x and y direction
    We use a star stencil for the finite difference discretization of the Laplacian operator. 
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,  \
                      DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,  \
                      1,1,NULL,NULL,&da);CHKERRQ(ierr); 
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);

  /* 
     Extract global vectors from DMDA, one for holding the rhs, one for the solution 
     and one for the manufactured solution; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &sol);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &b);CHKERRQ(ierr);

  /* 
    Create the matrix corresponding to the DMDA object.
  */
  DMCreateMatrix(da, &A);

  /*
    Compute the matrix and right-hand-side vector that define
    the linear system, Ax = b.
  */
  ierr = FDLaplaceOperatorDiscretize(da, &A);

  ierr = RhsAssembly(da, &b);CHKERRQ(ierr);

  /*   
    Set to 0 the solution vector (our initial guess).
  */
  ierr = VecSet(x,0.);CHKERRQ(ierr);


  /* 
     -----------------------------------------------
       solver section
     ----------------------------------------------- 
  */

   /*
      Create linear solver context
   */
   ierr = KSPCreate(PETSC_COMM_WORLD, &solver);CHKERRQ(ierr);

   /*
      Set the laplacian operator. Here the matrix that defines the linear system
      also serves as the preconditioning matrix.
   */
   ierr = KSPSetOperators(solver, A, A);CHKERRQ(ierr);

   /*
     Set runtime options, e.g.,
     -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.
   */
   ierr = KSPSetFromOptions(solver);CHKERRQ(ierr);

   /* 
     Solve the linear system
   */
   ierr = KSPSolve(solver,b,x);CHKERRQ(ierr);


  /* 
     -----------------------------------------------
       postprocessing section
     ----------------------------------------------- 
  */
   
  /*   
    Compute the manufactured solution.
  */
  ierr = ComputeManufacturedSolution(da, &sol);CHKERRQ(ierr);

  /* 
    Compute the relative error of the numerical solution 
    compared to the manufactured one and print it to the standard output.
  */
  VecNorm(sol,NORM_2,&norm_sol);
  VecAXPY(sol,-1.0,x);
  VecNorm(sol,NORM_2,&norm_diff);
  PetscPrintf(PETSC_COMM_WORLD, "l2 norm of the error: %g \n", (double)norm_diff/(double)norm_sol);

  /* 
    Print the solution and rhs in vtk format.
  */
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "output.vts", &viewer);
  PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);
  ierr= DMView(da,viewer);CHKERRQ(ierr);
  ierr= VecView(x,viewer);CHKERRQ(ierr);
  ierr= VecView(b,viewer);CHKERRQ(ierr);
  /* 
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  */

  /*
    Free work space.  All PETSc objects should be destroyed when they
    are no longer needed.
  */
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&solver);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /*   
    Finalize the program.
  */  
  ierr = PetscFinalize();
  
  PetscFunctionReturn(0);

}


/* 
  Evaluate the f function
  f= -32*(x*(x-1) + y*(y-1))
*/
double f(const double x, const double y) {
  double f_x = -32.*(x*(x-1.) + y*(y-1.));
  return f_x;
}


/* 
  Evaluate the sol function
  sol u = 16 * x * (x - 1) * y * (y - 1) 
*/
double u(const double x, const double y) {
  double sol =  16.*x*(x-1.)*y*(y-1.);
  return sol;
}


/* 
  Assemble the Matrix corresponding to the finite
  difference discretization of the Laplacian operator. 
*/
PetscErrorCode FDLaplaceOperatorDiscretize(DM dm, Mat* A)
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
PetscErrorCode RhsAssembly(DM dm, Vec* b) 
{
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  PetscScalar hx,hy;
  PetscInt i,j;  
  PetscScalar f_x;
  double x, y;

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
        x = (double)(hx*i);
        y = (double)(hy*j);
        f_x = f(x,y)*hx*hy;
        VecSetValue(*b, j+info.my*i, f_x, INSERT_VALUES);
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
  double x, y;
  PetscScalar val; 

  ierr  = DMDAGetLocalInfo(dm,&info); CHKERRQ(ierr);

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

  return 0;
}
