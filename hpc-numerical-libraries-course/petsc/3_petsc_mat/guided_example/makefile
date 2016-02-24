#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_mat

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_mat

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_mat: petsc_mat.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_mat petsc_mat.o  ${PETSC_KSP_LIB}
	${RM} petsc_mat.o
	@echo "-->...done.  <--"


