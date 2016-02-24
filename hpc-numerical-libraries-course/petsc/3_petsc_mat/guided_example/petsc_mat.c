static const char help[] = "Petsc mat example.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

   /* Initialize the Petsc environment */
   PetscInitialize(&argc,&argv,(char*)0,help);

   /*
     Write a PETSc program that creates a parallel matrix to host
     the 2d (5 point stencil) finite difference discretization of the 
     Laplace operator.
     Each processor needs to insert only elements that it owns locally
     (but any non-local elements will be sent to the appropriate processor
     during matrix assembly)
     Each processor prints the global and local size of the matrix
     nabla^2 f(x,y) = (f(x - h, y) + f(x + h, y) + f(x, y - h) + f(x, y + h) - 4 f(x,y))/h*h
   */

   PetscFinalize();

   PetscFunctionReturn(0);
}
