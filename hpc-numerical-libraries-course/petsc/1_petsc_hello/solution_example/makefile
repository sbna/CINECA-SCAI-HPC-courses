#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: 1_petsc_hello

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = 1_petsc_hello

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

1_petsc_hello: 1_petsc_hello.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 1_petsc_hello 1_petsc_hello.o  ${PETSC_KSP_LIB}
	${RM} 1_petsc_hello.o
	@echo "-->...done.  <--"


