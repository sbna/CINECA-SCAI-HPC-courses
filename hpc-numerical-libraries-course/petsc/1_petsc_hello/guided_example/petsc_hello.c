static const char help[] = "Petsc Hello World example.\n\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

   PetscInitialize(&argc,&argv,(char *)0,help);

   /* 
     Write a PETSc program where:
      - one processor prints on the std out the number of processors and its rank
      - all the processors print the string "Hello by proc <rank_of_the_proc>!" 
   */

   PetscFinalize();

   PetscFunctionReturn(0);

}


