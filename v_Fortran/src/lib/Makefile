GFC=gfortran -ffree-line-length-none

all: modrandom.o

modrandom.o: modrandom.f90
	$(GFC) -c modrandom.f90 -o modrandom.o && cp modrandom.mod ../
	

clean:
	@rm *~ &>/dev/null
	@rm *.o &>/dev/null
