#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_hello_solution

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_hello

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_hello_solution: petsc_hello_solution.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_hello_solution petsc_hello_solution.o  ${PETSC_KSP_LIB}
	${RM} petsc_hello_solution.o
	@echo "-->...done.  <--"


