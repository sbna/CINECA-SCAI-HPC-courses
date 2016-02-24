#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_mat_solution

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_mat_solution

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_mat_solution: petsc_mat_solution.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_mat_solution petsc_mat_solution.o  ${PETSC_KSP_LIB}
	${RM} petsc_mat_solution.o
	@echo "-->...done.  <--"
