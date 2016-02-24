CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(PETSC_TUTORIAL_GUIDED_EXAMPLE_4)

# configure the qsub script
CONFIGURE_FILE(
  "${PROJECT_SOURCE_DIR}/petsc_qsub_script.sh.in"
  "${PROJECT_BINARY_DIR}/petsc_qsub_script.sh" @ONLY IMMEDIATE
)

# Build the executable
ADD_EXECUTABLE(4_petsc_ksp_poisson  ${PROJECT_SOURCE_DIR}/4_petsc_ksp_poisson.c)

# Link the executable to the petsc and mpi libs
TARGET_LINK_LIBRARIES(4_petsc_ksp_poisson  ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})

