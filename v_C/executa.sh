#!/bin/bash
SERIE_ENTRADA="../../dados/dados_Africa/Af_22-25S.28-31E_1979-1999.bin"
MATRIZ_ENTRADA="../../dados/dados_AmSul/AmSul_1979-1999.bin"
ARQ_SAIDA="saida.bin"
NX=48
NY=56
NT=7665
#NT=2000
UNDEF=777.7
make && time ./mcarlo_seq $SERIE_ENTRADA $MATRIZ_ENTRADA $ARQ_SAIDA $NX $NY $NT $UNDEF

