#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: 2_petsc_vec

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = 2_petsc_vec

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

2_petsc_vec: 2_petsc_vec.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 2_petsc_vec 2_petsc_vec.o  ${PETSC_KSP_LIB}
	${RM} 2_petsc_vec.o
	@echo "-->...done.  <--"
