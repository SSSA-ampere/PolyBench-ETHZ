
# COMPILER OPTIONS -- ACCELERATOR
########################################

# Accelerator Compiler
ACC = clang

# Accelerator Compiler flags
ACCFLAGS+=-DNUM_TEAMS=1 -DNUM_THREADS=1024

ACC_INC_PATH+=-I.
ACC_LIB_PATH+=
ACC_LIBS+=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

# COMPILER OPTIONS -- HOST
########################################

# Compiler
CC = clang

# Compiler flags
CFLAGS = -O2

