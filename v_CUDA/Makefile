CC=gcc -Wall
FLAGS_CC=-lm
NVCC=nvcc
FLAGS_NVCC=-lm -lcudart -L/usr/lib/nvidia-current

SRC_MCARLO_SEQ=mcarlo_sequencial.c
SRC_MCARLO_CUDA=mcarlo_cuda.cu

tudo: mcarlo_cuda

mcarlo_cuda: declaracoes.h $(SRC_MCARLO_CUDA)
	@$(NVCC) $(SRC_MCARLO_CUDA) -o mcarlo_cuda $(FLAGS_NVCC)
	@echo "-> mcarlo_cuda"

mcarlo_seq: declaracoes.h $(SRC_MCARLO_SEQ)
	@$(CC) $(SRC_MCARLO_SEQ) -o mcarlo_seq $(FLAGS_CC)
	@echo "-> mcarlo_sequencial"
	
clean:
	@find . -iname "mcarlo_cuda" -exec rm {} \;
	@find . -iname "mcarlo_seq" -exec rm {} \;
