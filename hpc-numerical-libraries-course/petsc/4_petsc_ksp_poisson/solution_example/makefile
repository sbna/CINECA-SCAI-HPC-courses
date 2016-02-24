#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_ksp_poisson_solution

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_ksp_poisson_solution

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_ksp_poisson_solution: petsc_ksp_poisson_solution.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_ksp_poisson_solution petsc_ksp_poisson_solution.o  ${PETSC_KSP_LIB}
	${RM} petsc_ksp_poisson_solution.o
	@echo "-->...done.  <--"

