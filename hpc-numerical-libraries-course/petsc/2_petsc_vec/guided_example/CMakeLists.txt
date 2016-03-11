CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(PETSC_TUTORIAL_GUIDED_EXAMPLE_2)

# configure the qsub script
CONFIGURE_FILE(
  "${PROJECT_SOURCE_DIR}/petsc_qsub_script.sh"
  "${PROJECT_BINARY_DIR}/petsc_qsub_script.sh" @ONLY IMMEDIATE
)

# Build the executable
ADD_EXECUTABLE(petsc_vec  ${PROJECT_SOURCE_DIR}/petsc_vec.c)

# Link the executable to the petsc and mpi libs
TARGET_LINK_LIBRARIES(petsc_vec  ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})

