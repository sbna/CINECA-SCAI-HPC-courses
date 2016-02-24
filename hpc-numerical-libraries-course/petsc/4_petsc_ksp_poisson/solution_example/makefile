#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_ksp_poisson

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_ksp_poisson

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_ksp_poisson: petsc_ksp_poisson.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_ksp_poisson petsc_ksp_poisson.o  ${PETSC_KSP_LIB}
	${RM} petsc_ksp_poisson.o
	@echo "-->...done.  <--"

