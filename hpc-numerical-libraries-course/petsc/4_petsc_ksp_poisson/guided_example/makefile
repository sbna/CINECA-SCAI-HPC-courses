#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: 4_petsc_ksp_poisson

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = 4_petsc_ksp_poisson

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

4_petsc_ksp_poisson: 4_petsc_ksp_poisson.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 4_petsc_ksp_poisson 4_petsc_ksp_poisson.o  ${PETSC_KSP_LIB}
	${RM} 4_petsc_ksp_poisson.o
	@echo "-->...done.  <--"


