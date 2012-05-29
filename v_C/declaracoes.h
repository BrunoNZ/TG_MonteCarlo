#include <stdlib.h>
#include <stdio.h>
#include <errno.h>      
#include <string.h>
#include <math.h>
#include <time.h>

//OPCOES DO PROGRAMA
#define TOTAL_PERM 500 //Numero de pesos laterais

#define _250MB 262144000 // 250MB
#define _200MB 209715200 // 200MB
#define _150MB 157286400 // 150MB
#define _100MB 104857600 // 100MB
#define _50MB 52428800 // 50MB
#define _25MB 26214400 // 25MB
#define _15MB 15728640 // 15MB

//ESTRUTURA PARA ARMAZENAR OS PARAMETROS
//RELATIVOS AOS DADOS
typedef struct _parametros{
	char *arq_serie_entrada, *arq_matriz_entrada, *arq_saida;
	int NP, NT;
	float UNDEF;
	size_t TAM_MAX;
} parametros;
parametros param;

//ESTRUTURA PARA ARMAZENAR OS PARAMETROS
//RELATIVOS A EXECUCAO
typedef struct _parametros_exec{
	int npos_por_ciclo;
	int total_ciclos;
	int threads_por_bloco;
	int blocos_por_grid;
	size_t tam_por_ciclo;
} parametros_exec;
parametros_exec param_exec;

float *h_serie_entrada = NULL;
float *h_matriz_entrada = NULL;
float *h_saida = NULL;

float *d_serie_entrada = NULL;
float *d_matriz_entrada = NULL;
float *d_saida = NULL;
