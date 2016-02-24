#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: 3_petsc_mat

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = 3_petsc_mat

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

3_petsc_mat: 3_petsc_mat.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 3_petsc_mat 3_petsc_mat.o  ${PETSC_KSP_LIB}
	${RM} 3_petsc_mat.o
	@echo "-->...done.  <--"


