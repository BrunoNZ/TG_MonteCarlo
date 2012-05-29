CC=gcc -Wall
FLAGS_CC=-lm
NVCC=nvcc
FLAGS_NVCC=-lm -lcudart -L/usr/lib/nvidia-current

SRC_MCARLO_SEQ=mcarlo_sequencial.c
SRC_MCARLO_CUDA=mcarlo_cuda_v8.cu

SRC_MCARLO_SEQ_TESTE=mcarlo_seq_teste-mem.c

tudo: mcarlo_seq

teste: mcarlo_seq_teste

mcarlo_cuda: declaracoes.h $(SRC_MCARLO_CUDA)
	@$(NVCC) $(SRC_MCARLO_CUDA) -o mcarlo_cuda $(FLAGS_NVCC)
	@echo "-> mcarlo_cuda"

mcarlo_seq: declaracoes.h $(SRC_MCARLO_SEQ)
	@$(CC) $(SRC_MCARLO_SEQ) -o mcarlo_seq $(FLAGS_CC)
	@echo "-> mcarlo_sequencial"
	
mcarlo_seq_teste: declaracoes.h $(SRC_MCARLO_SEQ_TESTE)
	@$(CC) $(SRC_MCARLO_SEQ_TESTE) -o mcarlo_seq $(FLAGS_CC)
	@echo "-> mcarlo_seq_TESTE"	
	
clean:
	@find . -iname "lanczos_cuda" -exec rm {} \;
	@find . -iname "lanczos_seq" -exec rm {} \;
