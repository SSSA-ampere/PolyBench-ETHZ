INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)
OBJ = rose_$(BENCHMARK).c
SRC = $(BENCHMARK).c $(UTIL_DIR)/polybench.c
HEADERS = $(BENCHMARK).h $(UTIL_DIR)/polybench.h

DEPS        := Makefile.dep
DEP_FLAG    := -MM

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(OBJ) : $(SRC)
	$(ACC) $(ACCFLAGS) $(ACC_INC_PATH) $(INCPATHS) $^

$(EXE) : $(OBJ) $(BENCHMARK)-data.c 
	$(CC) -o $@ $(CFLAGS) $(ACC_INC_PATH) $(ACC_LIB_PATH) $(INCPATHS) $^ $(ACC_LIBS)

check: exe
	./$(EXE)

clean :
	-rm -vf __hmpp* -vf $(EXE) *~ 
	-rm -rf rose_$(BENCHMARK).c $(BENCHMARK)-data.c $(BENCHMARK).cl $(BENCHMARK)

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)
