#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_vec

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_vec

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_vec: petsc_vec.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_vec petsc_vec.o  ${PETSC_KSP_LIB}
	${RM} petsc_vec.o
	@echo "-->...done.  <--"
