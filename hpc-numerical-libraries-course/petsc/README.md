## Setup
Installation steps on GALILEO CINECA cluster

**clone the project**

    git clone https://github.com/sbna/CINECA-SCAI-HPC-courses.git
    
**load the environment**

    cd CINECA-SCAI-HPC-courses/hpc-numerical-libraries-course/petsc/
    source petsc_load_env.sh    

**configure the project**

    cd ../../../
    mkdir CINECA-SCAI-HPC-courses-bin
    cd CINECA-SCAI-HPC-courses-bin
    cmake ../CINECA-SCAI-HPC-courses/hpc-numerical-libraries-course
    
**compile the project**

    make

## Run an example
Open the petsc_qsub_script.sh script located in one of the example folders and change the last line of the script to run the desired example with custom flags

    cd petsc/X_petsc_xxx/solution_example
    qsub petsc_qsub_script.sh
