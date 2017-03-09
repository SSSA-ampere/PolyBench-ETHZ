APPNAME = reg_detect

HSA_CC    = /opt/rocm/gcc6/bin/gcc -march=znver1 --param hsa-expand-omp-parallel=1
NVIDIA_CC = 

ifdef SEQ
	OMPFLAGS = 
else
	OMPFLAGS = -fopenmp
endif

CC = ${HSA_CC}


INCDIR  = ../../utilities/
CFLAGS  = -O3 $(OMPFLAGS) -I$(INCDIR)  -DPOLYBENCH_TIME
LDFLAGS = -O3 $(OMPFLAGS) -lm

OUTDIR   = build
OBJFILES = polybench.o $(APPNAME).o
OBJ = $(patsubst %,$(OUTDIR)/%,$(OBJFILES))

.PHONY: clean

all: $(OBJ)
	$(CC) -o $(APPNAME)  $^ $(LDFLAGS)

$(OUTDIR)/%.o: %.c
	@mkdir -p $(OUTDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OUTDIR)/%.o: $(INCDIR)/%.c
	@mkdir -p $(OUTDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(OUTDIR) $(APPNAME)
