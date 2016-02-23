CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(HPC_NUMERICAL_LIBRARIES_COURSE)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")


#############################################################################################
### Compilers
#############################################################################################
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
IF(DEFINED CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
      build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build,
      options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()


#############################################################################################
### Find packages
#############################################################################################
# Find Petsc Library
FIND_PACKAGE(PETSc REQUIRED)
# Find MPI (optional)
FIND_PACKAGE(MPI)

#############################################################################################
### Include files
#############################################################################################
# Include petsc files
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})


#############################################################################################
### Add Tutorial Examples
#############################################################################################
# Add the petsc tutorial examples
ADD_SUBDIRECTORY(petsc)
