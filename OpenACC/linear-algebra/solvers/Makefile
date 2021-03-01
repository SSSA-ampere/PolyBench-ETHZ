
all:
	#make -C durbin all # not in OpenMP
	#make -C dynprog all # return zero
	make -C gramschmidt all
	make -C lu all
	#make -C ludcmp all # offloading error when running 

check:
	make -C durbin check
	make -C dynprog check
	make -C gramschmidt check
	make -C lu check
	make -C ludcmp check

clean:
	make -C durbin clean
	make -C dynprog clean
	make -C gramschmidt clean
	make -C lu clean
	make -C ludcmp clean

run:
	#durbin/a.out
	#dynprog/a.out
	gramschmidt/a.out
	lu/a.out
	#ludcmp/a.out
	