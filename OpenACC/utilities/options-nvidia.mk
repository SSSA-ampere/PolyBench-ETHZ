
# COMPILER OPTIONS -- ACCELERATOR
########################################

# Accelerator Compiler flags
CFLAGS+=-DNUM_TEAMS=1 -DNUM_THREADS=1024 -Wno-unknown-cuda-version
INCPATHS+=-I.
LIB_PATH+=
LIBS+=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

# COMPILER OPTIONS -- HOST
########################################

# Compiler
CC = clang

# Compiler flags
CFLAGS += -O2

