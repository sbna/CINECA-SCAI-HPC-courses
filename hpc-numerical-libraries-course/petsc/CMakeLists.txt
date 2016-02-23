CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(PETSC_TUTORIAL_EXAMPLES)

#############################################################################################
### Configuration project files
#############################################################################################
# configure the source env and qsub script
CONFIGURE_FILE(
  "${PROJECT_SOURCE_DIR}/petsc_load_env.sh"
  "${PROJECT_BINARY_DIR}/petsc_load_env.sh" @ONLY IMMEDIATE
)

CONFIGURE_FILE(
  "${PROJECT_SOURCE_DIR}/petsc_qsub_script.sh.in"
  "${PROJECT_BINARY_DIR}/petsc_qsub_script.sh" @ONLY IMMEDIATE
)

# Build the executables
ADD_EXECUTABLE(1_petsc_hello        ${PROJECT_SOURCE_DIR}/1_petsc_hello.c)
ADD_EXECUTABLE(2_petsc_vec          ${PROJECT_SOURCE_DIR}/2_petsc_vec.c)
ADD_EXECUTABLE(3_petsc_mat          ${PROJECT_SOURCE_DIR}/3_petsc_mat.c)
ADD_EXECUTABLE(4_petsc_ksp_poisson  ${PROJECT_SOURCE_DIR}/4_petsc_ksp_poisson.c)
ADD_EXECUTABLE(5_petsc_snes_poisson ${PROJECT_SOURCE_DIR}/5_petsc_snes_poisson.c)

# Link the executable to the petsc anf femttu libs
TARGET_LINK_LIBRARIES(1_petsc_hello        ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})
TARGET_LINK_LIBRARIES(2_petsc_vec          ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})
TARGET_LINK_LIBRARIES(3_petsc_mat          ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})
TARGET_LINK_LIBRARIES(4_petsc_ksp_poisson  ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})
TARGET_LINK_LIBRARIES(5_petsc_snes_poisson ${PETSC_LIBRARIES} ${MPI_EXTRA_LIBRARY})
