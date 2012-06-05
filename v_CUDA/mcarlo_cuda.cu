#include <cuda.h>
//#include "/opt/NVIDIA_GPU_Computing_SDK/C/src/simplePrintf/cuPrintf.cu"
#include "declaracoes.h"

/*------------------------------------------------------------------*/
/*	FUNCOES AUXILIARES												*/
/*------------------------------------------------------------------*/
int calcula_pos_matriz(int NT, int p, int t){
	return (p*NT)+t;
}

void le_argumentos(int argc, char **argv, parametros *param) {
	if (argc-1 != 8){
		printf("PARAMETROS INVALIDOS!\n");
		printf("Use: %s <ARQ_SERIE_ENTRADA> <ARQ_MATRIZ_ENTRADA> <ARQ_SAIDA> <NX> <NY> <NT> <UNDEF> <OP_TAM>\n",argv[0]);
		exit(1);
	}
	
	param->arq_serie_entrada=argv[1];
	param->arq_matriz_entrada=argv[2];
	param->arq_saida=argv[3];
	param->NP=atoi(argv[4])*atoi(argv[5]); //NP = NX * NY
	param->NT=atoi(argv[6]);
	param->UNDEF=atof(argv[7]);
	
	switch (atoi(argv[8])){  //argv[8] = OP_TAM 
		case 15 : param->TAM_MAX=(size_t)_15MB; break;
		case 25 : param->TAM_MAX=(size_t)_25MB; break;
		case 50 : param->TAM_MAX=(size_t)_50MB; break;
		case 100 : param->TAM_MAX=(size_t)_100MB; break;
		case 150 : param->TAM_MAX=(size_t)_150MB; break;
		case 200 : param->TAM_MAX=(size_t)_200MB; break;
		case 250 : param->TAM_MAX=(size_t)_250MB; break;
		default : printf("ERRO! Opcao de tamanho maximo invalida!\n"); exit(1);
	}
}

void le_serie_entrada(parametros param, float **s){
	FILE *arq;
	
	arq=fopen(param.arq_serie_entrada,"rb");
	if (!arq){
		printf("Erro na abertura do arquivo da serie de entrada : \"%s\".\n",param.arq_serie_entrada);
		exit (1);
	}
	
	(*s)=(float*)malloc((param.NT)*sizeof(float));
	
	fread((*s),sizeof(float),param.NT,arq);
	
	fclose(arq);
}

void le_matriz_entrada(parametros param, float **d){
	int p,t,pos;
	FILE *arq;
	float *buffer;

	arq=fopen(param.arq_matriz_entrada,"rb");
	if (!arq){
		printf("Erro na abertura do arquivo da matriz de entrada : \"%s\".\n",param.arq_matriz_entrada);
		exit (1);
	}
	
	buffer=(float*)malloc(param.NP*sizeof(float));

	(*d)=(float*)malloc((param.NP*param.NT)*sizeof(float));
	
	/*
	O arquivo binario esta organizado da seguinte maneira:
	[1,1,1][2,1,1][3,1,1],...,[NX,1,1]
	[1,2,1][2,2,1][3,2,1],...,[NX,2,1]
	[1,3,1][2,3,1][3,3,1],...,[NX,3,1]
	...
	[1,NY,1][2,NY,1][3,NY,1],...,[NX,NY,1]
	[1,1,2][2,1,2][3,1,2],...,[NX,1,2]
	[1,2,2][2,2,2][3,2,2],...,[NX,2,2]
	[1,3,2][2,3,2][3,3,2],...,[NX,3,2]
	...
	[1,NY,2][2,NY,2][3,NY,2],...,[NX,NY,2]
	[1,1,3][2,1,3][3,1,3],...,[NX,1,3]
	[1,2,3][2,2,3][3,2,3],...,[NX,2,3]
	[1,3,3][2,3,3][3,3,3],...,[NX,3,3]
	...
	[1,NY,2][2,NY,2][3,NY,NT-2],...,[NX,NY,NT-1]
	[1,1,3][2,1,3][3,1,NT-1],...,[NX,1,NT-1]
	[1,2,3][2,2,3][3,2,NT-1],...,[NX,2,NT-1]
	[1,3,3][2,3,3][3,3,NT-1],...,[NX,3,NT-1]
	...
	[1,NY,NT][2,NY,NT][3,NY,NT],...,[NX,NY,NT]

	Ou seja, varia varia primeiro o X, depois o Y, e por ultimo o T.
	*/
		
	for (t=0;t<param.NT;t++){
		fread(buffer,sizeof(float),param.NP,arq);

		for(p=0;p<param.NP;p++){
			pos=calcula_pos_matriz(param.NT,p,t);
			(*d)[pos]=buffer[p];
		}
	}

	free(buffer);
	
	fclose(arq);
}

void salva_arq_saida(parametros param, float *s){
	FILE *arq;
		
	arq=fopen(param.arq_saida,"wb");
	if (!arq){
		printf("Erro na abertura do arquivo de saida.\n");
		exit (1);
	}

	fwrite(s,sizeof(float),param.NP,arq);
		
	fclose(arq);	
}

void desaloca_variaveis(){
	if (h_serie_entrada != NULL) free(h_serie_entrada);
	if (h_matriz_entrada != NULL) free(h_matriz_entrada);
	if (h_saida != NULL) free(h_saida);
	
	if (d_serie_entrada != NULL) cudaFree(d_serie_entrada);
	if (d_matriz_entrada != NULL) cudaFree(d_matriz_entrada);
	if (d_saida != NULL) cudaFree(d_saida);
}

/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

/*------------------------------------------------------------------*/
/*	FUNCOES PARA DEBUGAR											*/
/*------------------------------------------------------------------*/
void imprime_dados_host(float *d, int p, int NT_IMPRESSAO, int NT){
	int t, ini_seq;
	ini_seq=(p*NT);
	for (t=0;t<NT_IMPRESSAO;t++)
		printf("[HOST] DADOS[%d][%d] (POS=%d) = %f\n",p,t,ini_seq+t,d[ini_seq+t]);

}
/*
__global__ void imprime_dados_device(const float *d, int p, int NT_IMPRESSAO, int NT){
	int t, ini_seq;
	ini_seq=(p*NT);
	for (t=200;t<NT_IMPRESSAO;t++)
		cuPrintf("[DEVICE] DADOS[%d][%d] (POS=%d) = %f\n",p,t,ini_seq+t,d[ini_seq+t]);
	
}
*/

/*
__global__ void imprime_dimensoes_execucao(){
	int p = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (blockIdx.x == 0 && threadIdx.x == 0)
		//cuPrintf("blockDim: %d / blockIdx: %d / threadIdx: %d / P: %d\n",
					blockDim.x,blockIdx.x,threadIdx.x,p);
}
*/

void imprime_argumentos(parametros param){
	printf("\n");
	printf("ARGUMENTOS:\n");
	printf(" > ARQ_SERIE_ENTRADA: %s\n",param.arq_serie_entrada);
	printf(" > ARQ_MATRIZ_ENTRADA: %s\n",param.arq_matriz_entrada);
	printf(" > ARQ_SAIDA: %s\n",param.arq_saida);
	printf(" > DIMENSOES: %d x %d\n",param.NP,param.NT);
	printf(" > UNDEF: %f\n",param.UNDEF);
	printf("\n");
}

void imprime_parametros_execucao(parametros_exec p_exec){
	printf("\n");
	printf("PARAMETROS DE EXECUCAO:\n");
    printf(" > TOTAL_POS = %d\n",p_exec.npos_por_ciclo);
    printf(" > TOTAL_CICLOS = %d\n",p_exec.total_ciclos);
    printf(" > THREADS_POR_BLOCO = %d\n",p_exec.threads_por_bloco);
    printf(" > BLOCOS_POR GRID = %d\n",p_exec.blocos_por_grid);
    printf(" > TAM_POR_CICLO = %ld\n",p_exec.tam_por_ciclo);
    printf("\n");
}
/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

/*------------------------------------------------------------------*/
/*	FUNCOES AUXILIARES ESPECIFICAS DO CUDA							*/
/*------------------------------------------------------------------*/
void verifica_erro_cuda(char *F_ID, int ciclo){
		cudaError_t erro = cudaGetLastError();
		if(cudaSuccess != erro){
			printf( "[CUDA_ERROR]: %s\n", cudaGetErrorString(erro) );
			printf( "\tID: %s (CICLO: %d)\n", F_ID, ciclo);
			desaloca_variaveis();
			exit(1);
		}
}

void calcula_parametros_execucao(parametros param, parametros_exec *param_exec){
									
	int npos_por_ciclo;
	int total_ciclos;
	size_t tam_por_ciclo;
	int threads_por_bloco;
	int blocos_por_grid;
	
	total_ciclos=1;
	npos_por_ciclo=param.NP;

	tam_por_ciclo=(npos_por_ciclo*param.NT)*sizeof(float);
	while ( tam_por_ciclo > param.TAM_MAX ){
		total_ciclos=total_ciclos+1;
		npos_por_ciclo=ceil(param.NP/total_ciclos);
		tam_por_ciclo=(npos_por_ciclo*param.NT)*sizeof(float);
	}
		
    threads_por_bloco = 8;
    blocos_por_grid = (npos_por_ciclo+threads_por_bloco-1)/threads_por_bloco;
           
    param_exec->npos_por_ciclo=npos_por_ciclo;
	param_exec->total_ciclos=total_ciclos;
	param_exec->tam_por_ciclo=tam_por_ciclo;
	param_exec->threads_por_bloco=threads_por_bloco;
	param_exec->blocos_por_grid=blocos_por_grid;

	return;
}
/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

/*------------------------------------------------------------------*/
/* FUNCOES PARA O CALCULO DA CORRELACAO E SIGNIFICANCIA				*/
/*------------------------------------------------------------------*/

__device__ void intrnd(int *seed){
	double const a=16807; //ie 7**5
	double const m=2147483647; //ie 2**31-1
	double const reciprocal_m=1.0/m;
	double temp = *seed * a;
	*seed = (int)(temp - m * floorf(temp * reciprocal_m));
}

__device__ void shuffle(float *array, size_t n, int seed){
	int rnd_number;
    if (n > 1) {
        size_t i;
        rnd_number=seed; //0 nao funciona
		for (i = 0; i < n - 1; i++) {
			intrnd(&rnd_number);
			size_t j = i + rnd_number / (RAND_MAX / (n - i) + 1);
			float t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
    }
}

__device__ float correlacao_serie_serie(float *vA, float *vB, int NT, float undef){
	
	int i,k;
	float E_cov, E_anm2_A, E_anm2_B, div;
	float med_A, med_B;
	float res;
	
	//*** k = Numero de duplas validas e tamanho total das series auxiliares ***
	med_A=0.0;
	med_B=0.0;
	k=0;
	for (i=0;i<NT;i++){
		//if ((vA[i] != undef) && (vB[i] != undef)){
			med_A=med_A+vA[i];
			med_B=med_B+vB[i];
					
			k++;
		//}
	}
	if (k < 2) return undef;
		
	med_A=med_A/k;
	med_B=med_B/k;
	E_cov=0.0;
	E_anm2_A=0.0;
	E_anm2_B=0.0;
	for (i=0;i<NT;i++){
		//if ((vA[i] != undef) && (vB[i] != undef)){
			E_cov    = E_cov	+ ((vA[i]-med_A)*(vB[i]-med_B));
			E_anm2_A = E_anm2_A	+ ((vA[i]-med_A)*(vA[i]-med_A));
			E_anm2_B = E_anm2_B	+ ((vB[i]-med_B)*(vB[i]-med_B));
		//}
	}
	
	div=sqrtf(E_anm2_A * E_anm2_B);
	if (div == 0) res=undef;
	else res=E_cov/div;
	
	return res;
}

__global__ void sig_mcarlo_serie_serie(
					float *vA, float *vB, int NT, float undef, int total_perm,
					int total_pos, float *saida){
	
	int i, ini_seq, cont;
	float correl_orig, correl;
	float correl_orig_abs;
	
	int p = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (p >= total_pos-1) return;
	
	ini_seq=(p*NT);
	
	correl_orig=correlacao_serie_serie(vA,(vB+ini_seq),NT,undef);
	correl_orig_abs=fabsf(correl_orig);

	if (correl_orig == undef){
		saida[p]=undef;
		return;
	}
	
	cont=0;
	for (i=0;i<total_perm;i++){
		shuffle((vB+ini_seq),NT,(p+i)+1);
		correl=fabsf(correlacao_serie_serie(vA,(vB+ini_seq),NT,undef));
		//if (fabsf(correl) >= correl_orig_abs) cont++;
		cont=cont+(correl >= correl_orig_abs);
	}
		
	saida[p]=((float)cont/(float)total_perm)*(correl_orig*fabsf(correl_orig));
	
	return;
}

/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

int main(int argc, char **argv){
	
	int total_npos, ciclo;
	int pos_inicio_copia_entrada, pos_inicio_copia_saida;

	//LE OS ARGUMENTOS
	le_argumentos(argc,argv,&param);
	imprime_argumentos(param);
	
	//LE OS DADOS DE ENTRADA
	le_serie_entrada(param,&h_serie_entrada);
	le_matriz_entrada(param,&h_matriz_entrada);
	
	//****************************************************************************//
	//EXECUCAO CUDA
	
	//CALCULA OS PARAMETROS DE EXECUCAO
	calcula_parametros_execucao(param, &param_exec);
	imprime_parametros_execucao(param_exec);
	
	//ALOCA E COPIA A SERIE DE DADOS DE ENTRADA
	cudaMalloc((void**)&d_serie_entrada, param.NT*sizeof(float) );
		verifica_erro_cuda("cudaMalloc(d_serie_entrada)",-1);
	cudaMemcpy(d_serie_entrada,h_serie_entrada,param.NT*sizeof(float), cudaMemcpyHostToDevice);
		verifica_erro_cuda("cudaMemcpy(d_serie_entrada)",-1);
	
	//ALOCA O ESPACO PARA OS DADOS DE ENTRADA E SAIDA NO DEVICE
	cudaMalloc((void**)&d_matriz_entrada, param_exec.tam_por_ciclo );
		verifica_erro_cuda("cudaMalloc(d_matriz_entrada)",-1);
	cudaMalloc((void**)&d_saida, param_exec.npos_por_ciclo*sizeof(float) );
		verifica_erro_cuda("cudaMalloc(d_saida)",-1);
	
	//CALCULA O NUMERO TOTAL DE POSICOES QUE SERAO CALCULADAS
	// *** ESSE NUMERO PROVAVELMENTE SERA MAIOR QUE O NP,
	// *** POR ISSO NAO SE PODE ALOCAR EXATAMENTE O TAMANHO DA SAIDA ESPERADO.
	// *** TEM QUE ALOCAR ESSA MEMORIA A MAIS PARA GARANTIR
	// *** QUE NO ULTIMO CICLO NAO SEJA COPIADO DADOS ALEM DO QUE FOI ALOCADO
	total_npos=param_exec.npos_por_ciclo*param_exec.total_ciclos;
	
	//ALOCA NO HOST O ESPACO PARA A SAIDA (TOTAL)
	h_saida=(float*)malloc(total_npos*sizeof(float));
	
	//printf(" --> EXECUTANDO CICLO:"); fflush(stdout);
	for (ciclo=0;ciclo<param_exec.total_ciclos;ciclo++){
		
		//RESETA A MEMORIA DO DEVICE
		//cudaMemset(d_matriz_entrada,0,param_exec.tam_por_ciclo);
		//	verifica_erro_cuda("cudaMemset(d_matriz_entrada)",ciclo);
		//cudaMemset(d_saida,0,param_exec.npos_por_ciclo*sizeof(float));
		//	verifica_erro_cuda("cudaMemset(d_saida)",ciclo);

		//CALCULA A POSICAO DE INICIO DA COPIA DOS DADOS DE ENTRADA
		pos_inicio_copia_entrada=(ciclo*param_exec.npos_por_ciclo)*param.NT;
				
		//COPIA OS DADOS DA MATRIZ DE ENTRADA
		cudaMemcpy(d_matriz_entrada,(h_matriz_entrada+pos_inicio_copia_entrada),
				param_exec.tam_por_ciclo, cudaMemcpyHostToDevice);
			verifica_erro_cuda("cudaMemcpy(d_matriz_entrada)",ciclo);

		//EXECUTA O MCARLO EM CUDA
		sig_mcarlo_serie_serie<<<param_exec.blocos_por_grid, param_exec.threads_por_bloco>>>(
				d_serie_entrada, d_matriz_entrada, param.NT, param.UNDEF, TOTAL_PERM,
				param_exec.npos_por_ciclo, d_saida);
			verifica_erro_cuda("sig_mcarlo_serie_serie",ciclo);

		//ESPERA O TERMINIO DA EXECUCAO DO DEVICE
		cudaDeviceSynchronize();
			verifica_erro_cuda("cudaDeviceSynchronize",ciclo);
		
		//CALCULA A POSICAO DE INICIO DA COPIA DOS DADOS DE SAIDA
		pos_inicio_copia_saida=ciclo*param_exec.npos_por_ciclo;
		
		//COPIA A SAIDA DO DEVICE PARA O HOST
		cudaMemcpy((h_saida+pos_inicio_copia_saida),d_saida,
				param_exec.npos_por_ciclo*sizeof(float), cudaMemcpyDeviceToHost);
			verifica_erro_cuda("cudaMemcpy(h_saida)",ciclo);
	}
	//****************************************************************************//

	//for (int p=0;p<param.NP;p++) if (h_saida[p] != param.UNDEF) printf("%f\n",h_saida[p]);
	
	salva_arq_saida(param, h_saida);
	
	desaloca_variaveis();
	
	printf("FIM!\n");
		
	return 0;
}
