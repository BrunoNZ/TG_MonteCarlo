GFC=gfortran -ffree-line-length-none
SRC=./src
NOME=correlacao
LIB_MODRANDOM=$(SRC)/lib/modrandom.o

all: correlacao

correlacao: $(LIB_MODRANDOM) $(SRC)/$(NOME).f90
	$(GFC) -o $(NOME) $(LIB_MODRANDOM) $(SRC)/$(NOME).f90

clean:
	@find . -name $(NOME) -exec rm {} \;
	@find $(SRC) -maxdepth 0 -iname "*~" -exec rm {} \;
	@find $(SRC) -maxdepth 0 -iname "*.o" -exec rm {} \;
	@echo "Tudo limpo!"
