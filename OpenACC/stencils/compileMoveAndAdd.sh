#!/bin/bash

INSTALL_PATH=$1
NAME_PREFIX=$2
COMMENT=$3

[[  -z  $INSTALL_PATH  ]] && echo "No install path!" && exit
[[  -z  $NAME_PREFIX  ]] && echo "No prefix!" && exit
[[  -z  $COMMENT  ]] && echo "No comment!" && exit

./compileAndMove.sh "$INSTALL_PATH" "$NAME_PREFIX"
./addToTargets.sh "$INSTALL_PATH" "$COMMENT"
