CC=gcc -lm -Wall
NVCC=nvcc -lm -lcudart -L/usr/lib/nvidia-current

SRC_MCARLO_SEQ=mcarlo_sequencial.c
SRC_MCARLO_CUDA=mcarlo_cuda_v8.cu

tudo: mcarlo_seq

#mcarlo_cuda: declaracoes.h $(SRC_MCARLO_CUDA)
#	@$(NVCC) $(SRC_MCARLO_CUDA) -o mcarlo_cuda
#	@echo "-> mcarlo_cuda"

mcarlo_seq: declaracoes.h $(SRC_MCARLO_SEQ)
	@$(CC) $(SRC_MCARLO_SEQ) -o mcarlo_seq
	@echo "-> mcarlo_sequencial"
	
clean:
	@find . -iname "lanczos_cuda" -exec rm {} \;
	@find . -iname "lanczos_seq" -exec rm {} \;
