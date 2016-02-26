/**
 * @file 5_petsc_snes_poisson.c
 * @author Simone Bnà
 * @date 19 Feb 2016
 * @brief File containing the example of the solution in parallel 
 * of a Poisson problem using SNES and DMDA.
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
 * source loadPetscEnv.sh 
 * make
 * qsub petscSubmissionScript
 */

static const char help[] = "-Laplacian u = b as a nonlinear problem.\n\n";

/*

    The linear and nonlinear versions of these should give almost identical results on this problem

    Richardson
      Nonlinear:
        -snes_rtol 1.e-12 -snes_monitor -snes_type nrichardson -snes_linesearch_monitor

      Linear:
        -snes_rtol 1.e-12 -snes_monitor -ksp_rtol 1.e-12  -ksp_monitor -ksp_type richardson -pc_type none -ksp_richardson_self_scale -info

    GMRES
      Nonlinear:
       -snes_rtol 1.e-12 -snes_monitor  -snes_type ngmres

      Linear:
       -snes_rtol 1.e-12 -snes_monitor  -ksp_type gmres -ksp_monitor -ksp_rtol 1.e-12 -pc_type none

    CG
       Nonlinear:
            -snes_rtol 1.e-12 -snes_monitor  -snes_type ncg -snes_linesearch_monitor

       Linear:
             -snes_rtol 1.e-12 -snes_monitor  -ksp_type cg -ksp_monitor -ksp_rtol 1.e-12 -pc_type none

    Multigrid
       Linear:
          1 level:
            -snes_rtol 1.e-12 -snes_monitor  -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type none -mg_levels_ksp_monitor
            -mg_levels_ksp_richardson_self_scale -ksp_type richardson -ksp_monitor -ksp_rtol 1.e-12  -ksp_monitor_true_residual

          n levels:
            -da_refine n

       Nonlinear:
         1 level:
           -snes_rtol 1.e-12 -snes_monitor  -snes_type fas -fas_levels_snes_monitor

          n levels:
            -da_refine n  -fas_coarse_snes_type newtonls -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly

*/



/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormMatrix(DM,Mat);
extern PetscErrorCode MyComputeFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode MyComputeJacobian(SNES,Vec,Mat,Mat,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;        /* nonlinear solver */
  Vec            x,b;         /* solution vector */
  PetscInt       its;         /* iterations for convergence */
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    viewer;

  /*
    Initialize program
  */
  PetscInitialize(&argc,&argv,(char*)0,help);

  /*
    Create distributed array (DMDA) to manage parallel grid and vectors
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,\
                      DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,\
                      NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);

  /*
    Extract global vectors from DMDA; then duplicate for remaining
    vectors that are the same types
  */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecSet(b,0.);CHKERRQ(ierr);

  /* 
    Create nonlinear solver context (mesh, compute rhs and jacobian functions)
  */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,NULL,MyComputeFunction,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,MyComputeJacobian,NULL);CHKERRQ(ierr);

  /*
    Customize nonlinear solver; set runtime options
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /*
    Solve nonlinear system
  */
  ierr = SNESSolve(snes,b,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  /*
    Print the mesh and solution in vtk format
  */
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "output.vts", &viewer);
  PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);
  ierr= DMView(da,viewer);CHKERRQ(ierr);
  ierr= VecView(x,viewer);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MyComputeFunction"
PetscErrorCode MyComputeFunction(SNES snes,Vec x,Vec F,void *ctx)
{
  DM dm;
  DMDALocalInfo  info;   
  Mat J;
  PetscScalar hx,hy;
  PetscInt i,j;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm,&J);CHKERRQ(ierr);
  if (!J) {
    ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
    ierr = MatSetDM(J, NULL);CHKERRQ(ierr);
    ierr = FormMatrix(dm,J);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(dm,J);CHKERRQ(ierr);
    ierr = DMSetApplicationContextDestroy(dm,(PetscErrorCode (*)(void**))MatDestroy);CHKERRQ(ierr);
  }
  ierr = MatMult(J,x,F);CHKERRQ(ierr);

  ierr  = DMDAGetLocalInfo(dm,&info); CHKERRQ(ierr);
  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);

  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {     
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        VecSetValue(F, j+info.my*i, 0., ADD_VALUES);
      }
      else {
        PetscScalar f = 32.*( hx*i*(hx*i - 1.) + hy*j*(hy*j - 1.) )*hx*hy;
        VecSetValue(F, j+info.my*i, f, ADD_VALUES);
      }
    }
  }

  VecAssemblyBegin(F);
  VecAssemblyEnd(F);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyComputeJacobian"
PetscErrorCode MyComputeJacobian(SNES snes,Vec x,Mat J,Mat Jp,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = FormMatrix(dm,Jp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormMatrix"
PetscErrorCode FormMatrix(DM da,Mat jac)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nrows = 0;
  MatStencil     col[5],row,*rows;
  PetscScalar    v[5],hx,hy,hxdhy,hydhx;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  ierr  = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
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
        ierr            = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        rows[nrows].i   = i;
        rows[nrows++].j = j;
      } else {
        /* interior grid points */
        v[0] = -hxdhy;                                           col[0].j = j - 1; col[0].i = i;
        v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
        v[2] = 2.0*(hydhx + hxdhy);                              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
        v[4] = -hxdhy;                                           col[4].j = j + 1; col[4].i = i;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatZeroRowsColumnsStencil(jac,nrows,rows,2.0*(hydhx + hxdhy),NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

