#!/bin/bash

INSTALL_PATH_BASE="/home/bjoernf/scratch/PolyBench-ACC/OpenACC/stencils/hercules-bin"
FROM_DIR=$1
COMMENT=$2

[[  -z  $FROM_DIR  ]] && echo "No from path!" && exit
[[  -z  $COMMENT  ]] && echo "No comment!" && exit

TARGETS_FILE="/home/bjoernf/scratch/gguard-lkm/autorun/bulk-overrun/targets.txt"

cd $INSTALL_PATH_BASE/$FROM_DIR

echo "# $COMMENT" >> $TARGETS_FILE

for f in *
do 
    echo "$INSTALL_PATH_BASE/$FROM_DIR/$f" >> $TARGETS_FILE
done
