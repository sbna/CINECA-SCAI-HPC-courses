static const char help[] = "-Laplacian u = b as a linear problem.\n\n";

/*T
   Concepts: KSP^parallel Poisson example
   Concepts: DMDA^using distributed arrays;
   Processors: n
T*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>  

int main(int argc,char **argv)
{
  Vec            x,b,sol;                                  /* solution vector */
  PetscInt       its;                                  /* iterations for convergence */
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    viewer;
  KSP            solver;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char*)0,help);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,    DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecSet(x,0.);CHKERRQ(ierr);


  /*
    Assemble the matrix

  */

  PetscInt       i,j,nrows = 0;
  MatStencil     col[5],row,*rows;
  PetscScalar    v[5],hx,hy,hxdhy,hydhx;
  DMDALocalInfo  info;
  Mat A;

  DMCreateMatrix(da, &A);

  PetscFunctionBeginUser;
  ierr  = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);
  hxdhy = hx/hy;
  hydhx = hy/hx;

  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  printf("Number of processors = %d, rank = %d\n",size,rank);

  printf("rank: %d  mxa mxb: %d, %d   mya myb: %d  %d\n", rank, info.ys, info.ys+info.ym, info.xs, info.xs+info.xm);
  /*PetscPrintf(PETSC_COMM_SELF, "mxa mxb: %d, %d   mya myb: %d  %d\n", info.ys, info.ys+info.ym, info.xs, info.xs+info.xm);*/

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
        ierr            = MatSetValuesStencil(A,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        rows[nrows].i   = i;
        rows[nrows++].j = j;
      } else {
        /* interior grid points */
        v[0] = -hxdhy;                                           col[0].j = j - 1; col[0].i = i;
        v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
        v[2] = 2.0*(hydhx + hxdhy);                              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
        v[4] = -hxdhy;                                           col[4].j = j + 1; col[4].i = i;
        ierr = MatSetValuesStencil(A,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatZeroRowsColumnsStencil(A,nrows,rows,2.0*(hydhx + hxdhy),NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  
  /* compute rhs */
  /*PetscScalar f = 1.*hx*hy;
  ierr = VecSet(b,f);CHKERRQ(ierr);*/

  /* bdc for b */
  for (int j=info.ys; j<info.ys+info.ym; j++) {
    for (int i=info.xs; i<info.xs+info.xm; i++) {     
      /* boundary points */
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        VecSetValue(b, j+info.my*i, 0., INSERT_VALUES);
      }
      else {
        PetscScalar f = -32.*( hx*i*(hx*i - 1.) + hy*j*(hy*j - 1.) )*hx*hy;
        VecSetValue(b, j+info.my*i, f, INSERT_VALUES);
      }
    }
  }

  VecAssemblyBegin(b);
  VecAssemblyEnd(b);




  /* 

    Solution part

  */

   /*
      Create linear solver context
   */
   KSPCreate(PETSC_COMM_WORLD, &solver);

   /*
      Set operators. Here the matrix that defines the linear system
      also serves as the preconditioning matrix.
   */
   KSPSetOperators(solver, A, A);

   /*
     Set runtime options, e.g.,
     -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.
   */
   KSPSetFromOptions(solver);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   KSPSolve(solver,b,x);


   /*
      Create the manufactured solution. Check the l2norm of the error. 

   */
   ierr = VecDuplicate(x, &sol);
   for (int j=info.ys; j<info.ys+info.ym; j++) {
     for (int i=info.xs; i<info.xs+info.xm; i++) {     
        PetscScalar f = 16.*(hx*i)*(hx*i - 1.)*(hy*j)*(hy*j - 1.);
        VecSetValue(sol, j+info.my*i, f, INSERT_VALUES);
     }
   }

  VecAssemblyBegin(sol);
  VecAssemblyEnd(sol);

  PetscReal norm_sol;
  VecNorm(sol,NORM_2,&norm_sol);

  VecAXPY(sol,-1.0,x);
  PetscReal norm_diff;
  VecNorm(sol,NORM_2,&norm_diff);

  PetscPrintf(PETSC_COMM_WORLD, "l2 norm of the error: %g \n", (double)norm_diff/(double)norm_sol);

   /* 
    
   View section

  */

  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "testing.vtk", &viewer);
  PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);
  ierr= DMView(da,viewer);CHKERRQ(ierr);
  ierr= VecView(x,viewer);CHKERRQ(ierr);
  ierr= VecView(b,viewer);CHKERRQ(ierr);
  /*ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);*/

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&solver);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
ierr = PetscFinalize();
  
  PetscFunctionReturn(0);

}

