#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: 5_petsc_snes_poisson

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = 5_petsc_snes_poisson

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

5_petsc_snes_poisson: 5_petsc_snes_poisson.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 5_petsc_snes_poisson 5_petsc_snes_poisson.o  ${PETSC_KSP_LIB}
	${RM} 5_petsc_snes_poisson.o
	@echo "-->...done.  <--"
