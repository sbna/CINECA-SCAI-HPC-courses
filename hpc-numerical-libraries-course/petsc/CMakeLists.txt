CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(PETSC_TUTORIAL_EXAMPLES)

#############################################################################################
### Add Tutorial Examples
#############################################################################################
# Add the petsc tutorial examples
ADD_SUBDIRECTORY(1_petsc_hello)
ADD_SUBDIRECTORY(2_petsc_vec)
ADD_SUBDIRECTORY(3_petsc_mat)
ADD_SUBDIRECTORY(4_petsc_ksp_poisson)
ADD_SUBDIRECTORY(5_petsc_snes_poisson)

