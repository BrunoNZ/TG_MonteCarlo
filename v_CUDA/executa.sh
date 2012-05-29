#!/bin/bash
SERIE_ENTRADA="/home/bruno/Materias_UFPR/TG_MonteCarlo/dados/Af_22-25S.28-31E_1979-1999.bin"
MATRIZ_ENTRADA="/home/bruno/Materias_UFPR/TG_MonteCarlo/dados/AmSul_1979-1999.bin"
ARQ_SAIDA="saida.bin"
NX=48
NY=56
NT=7665
UNDEF=777.7
make && time ./mcarlo_cuda $SERIE_ENTRADA $MATRIZ_ENTRADA $ARQ_SAIDA $NX $NY $NT $UNDEF 25

