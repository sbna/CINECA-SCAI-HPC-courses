PETSC_DIR  = /home/ubuntu/Software/petsc
PETSC_ARCH = arch-linux2-cxx-opt

ALL: 1_petsc_hello 2_petsc_vec 3_petsc_mat 4_petsc_ksp_poisson
#ALL: 3_petsc_mat

#CFLAGS   = -g -O2
CFLAGS = -g -O2
CPPFLAGS = 

CLEANFILES = 1_petsc_hello 2_petsc_vec 3_petsc_mat 4_petsc_ksp_poisson

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

1_petsc_hello: 1_petsc_hello.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 1_petsc_hello 1_petsc_hello.o  ${PETSC_KSP_LIB}
	${RM} 1_petsc_hello.o
	@echo "-->...done.  <--"

2_petsc_vec: 2_petsc_vec.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 2_petsc_vec 2_petsc_vec.o  ${PETSC_KSP_LIB}
	${RM} 2_petsc_vec.o
	@echo "-->...done.  <--"

3_petsc_mat: 3_petsc_mat.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 3_petsc_mat 3_petsc_mat.o  ${PETSC_KSP_LIB}
	${RM} 3_petsc_mat.o
	@echo "-->...done.  <--"

4_petsc_ksp_poisson: 4_petsc_ksp_poisson.o  chkopts
	@echo "--> Linking the executable...  <--"
	-${CLINKER} -o 4_petsc_ksp_poisson 4_petsc_ksp_poisson.o  ${PETSC_KSP_LIB}
	${RM} 4_petsc_ksp_poisson.o
	@echo "-->...done.  <--"
