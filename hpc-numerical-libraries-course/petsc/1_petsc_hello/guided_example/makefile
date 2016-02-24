#PETSC_DIR  = ""
#PETSC_ARCH = ""

ALL: petsc_hello

CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = petsc_hello

#include ${PETSC_HOME}/conf/variables
#include ${PETSC_HOME}/conf/rules
include ${PETSC_HOME}/lib/petsc/conf/variables
include ${PETSC_HOME}/lib/petsc/conf/rules

petsc_hello: petsc_hello.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o petsc_hello petsc_hello.o  ${PETSC_KSP_LIB}
	${RM} petsc_hello.o
	@echo "-->...done.  <--"


