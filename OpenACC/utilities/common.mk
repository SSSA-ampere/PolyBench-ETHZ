INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)
SRC = $(BENCHMARK).c $(UTIL_DIR)/polybench.c
HEADERS = $(BENCHMARK).h $(UTIL_DIR)/polybench.h

DEPS        := Makefile.dep
DEP_FLAG    := -MM

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(CFLAGS) $(ACCFLAGS)  $(ACC_INC_PATH) $(ACC_LIB_PATH) $(INCPATHS) $^ $(ACC_LIBS)

check: exe
	./$(EXE)

clean :
	-rm -rf __hmpp* $(EXE) a.out *~ 
	-rm -rf Makefile.dep

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)
