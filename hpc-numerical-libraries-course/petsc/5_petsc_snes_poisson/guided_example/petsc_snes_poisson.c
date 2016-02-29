static const char help[] = "-Laplacian u = b as a nonlinear problem.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormMatrix(DM,Mat);
extern PetscErrorCode MyComputeFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode MyComputeJacobian(SNES,Vec,Mat,Mat,void*);
double f(const double x, const double y);


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
    Use the SNESSetFunction and SNESSetJacobian together with 
    FormMatrix, MyComputeFunction and MyComputeJacobian 
    to evaluate the jacobian J and the non linear function F(x) 
    in a SNES context.
    Solve the problem with a SNES object.
  */

  PetscFinalize();
  
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
  double xx, xy;
  PetscScalar val;

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
        xx = (double)(hx*i);
        xy = (double)(hx*j);
        val = -1.*f(xx, xy)*hx*hy;
        VecSetValue(F, j+info.my*i, val, ADD_VALUES);
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

