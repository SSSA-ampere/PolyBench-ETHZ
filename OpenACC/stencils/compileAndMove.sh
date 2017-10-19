#!/bin/bash

# $1 = Install Path
# $2 = Prefix
# $3 = Suffix
INSTALL_PATH=$1
NAME_PREFIX=$2

[[  -z  $INSTALL_PATH  ]] && echo "No install path!" && exit
[[  -z  $NAME_PREFIX  ]] && echo "No prefix!" && exit

STENCIL_PATH="/home/bjoernf/scratch/PolyBench-ACC/OpenACC/stencils"
VECTOR_PATH="/home/bjoernf/scratch/OpenMP-Examples/vectorAdd"
INSTALL_PATH_BASE="/home/bjoernf/scratch/PolyBench-ACC/OpenACC/stencils/hercules-bin"

#
# REMEMBER WHERE HERE IS
#
ORIG_PATH=$(pwd)

#
# STENCILS
# 
declare -a STENCILS=("adi" "convolution-2d" "fdtd-2d" "jacobi-2d-imper" "seidel-2d")
for STENCIL in "${STENCILS[@]}" 
do
   cd "$STENCIL_PATH/$STENCIL"
   clang -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xcuda-ptxas -maxrregcount=32 -O1 $STENCIL.c ../../utilities/polybench.c -I../../../common -DNUM_TEAMS=1 -DNUM_THREADS=1024
   cp a.out "$INSTALL_PATH_BASE/$INSTALL_PATH/$NAME_PREFIX-$STENCIL"
done

#
# VECTORADD
#
cd "$VECTOR_PATH"
clang -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xcuda-ptxas -maxrregcount=32 -O1 vectorAdd.omp.cpp
cp a.out "$INSTALL_PATH_BASE/$INSTALL_PATH/$NAME_PREFIX-vectorAdd"

#
# RETURN HERE
#
echo "Returning to Original Path: $ORIG_PATH"
cd "$ORIG_PATH"
