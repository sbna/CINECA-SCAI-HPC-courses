static const char help[] = "Petsc vec example.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

  /* Initialize the Petsc environment */
  PetscInitialize(&argc,&argv,(char*)0,help);
  
  /* Write a petsc program that creates a parallel vector and fills it 
     according to the formula v[i] = i
     in three different ways:
      - each processor fills all the entries
      - each processor fills only its local entries using VecSetValue
      - each processor fills only its local entries using VecGetArray 
      - each processor prints the global and local size of the vector 
  */

  PetscFinalize();

  PetscFunctionReturn(0);
}
