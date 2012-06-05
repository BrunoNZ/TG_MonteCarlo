#include "declaracoes.h"

/*------------------------------------------------------------------*/
/*	FUNCOES AUXILIARES												*/
/*------------------------------------------------------------------*/
int calcula_pos_matriz(int NT, int p, int t){
	return (p*NT)+t;
}

void le_argumentos(int argc, char **argv, parametros *param) {
	if (argc-1 != 7){
		printf("PARAMETROS INVALIDOS!\n");
		printf("Use: %s <ARQ_SERIE_ENTRADA> <ARQ_MATRIZ_ENTRADA> <ARQ_SAIDA> <NX> <NY> <NT> <UNDEF>\n",argv[0]);
		exit(1);
	}
	param->arq_serie_entrada=argv[1];
	param->arq_matriz_entrada=argv[2];
	param->arq_saida=argv[3];
	param->NP=atoi(argv[4])*atoi(argv[5]); //NP = NX * NY
	param->NT=atoi(argv[6]);
	param->UNDEF=atof(argv[7]);
}

void imprime_argumentos(parametros param){
	printf("\n");
	printf("ARGUMENTOS:\n");
	printf("-> ARQ_SERIE_ENTRADA: %s\n",param.arq_serie_entrada);
	printf("-> ARQ_MATRIZ_ENTRADA: %s\n",param.arq_matriz_entrada);
	printf("-> ARQ_SAIDA: %s\n",param.arq_saida);
	printf("-> DIMENSOES: %d x %d\n",param.NP,param.NT);
	printf("-> UNDEF: %f\n",param.UNDEF);
	printf("\n");
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
		printf("Erro na abertura do arquivo da serie de entrada : \"%s\".\n",param.arq_matriz_entrada);
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

void shuffle(float *array, size_t n){
    if (n > 1) {
        size_t i;
		for (i = 0; i < n - 1; i++) {
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			float t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
    }
}

/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

float correlacao_serie_serie(float *vA, float *vB, int nt, float undef){
	
	int i,k;
	float *vA_aux, *vB_aux;
	double E_cov, E_anm2_A, E_anm2_B, div;
	double med_A, med_B;
		
	vA_aux=(float *)malloc(nt*sizeof(float));
	vB_aux=(float *)malloc(nt*sizeof(float));
	
	//CRIA DOIS VETORES AUXILIARES APENAS COM AS "DUPLAS" DE
	//DADOS ONDE AMBOS OS VALORES NAO SEJAM INDEFINIDOS.
	//ALEM DISSO JA FAZ O SOMATORIO DAS SERIES FINAIS PARA DEPOIS
	//CALCULAR A MEDIA;
	//*** k = Numero de duplas validas e tamanho total das series auxiliares ***
	med_A=0.0;
	med_B=0.0;
	k=0; 
	for (i=0;i<nt;i++){
		if ((vA[i] != undef) && (vB[i] != undef)){
			vA_aux[k]=vA[i];
			vB_aux[k]=vB[i];
			
			med_A=med_A+(double)vA[i];
			med_B=med_B+(double)vB[i];
						
			k++;
		}
	}
	if (k < 2) {
		free(vA_aux);
		free(vB_aux);
		return (double)undef;
	}
		
	med_A=med_A/(double)k;
	med_B=med_B/(double)k;
	E_cov=0.0;
	E_anm2_A=0.0;
	E_anm2_B=0.0;
	for (i=0;i<k;i++){
		E_cov    = E_cov	+ (((double)vA_aux[i]-med_A)*((double)vB_aux[i]-med_B));
		E_anm2_A = E_anm2_A	+ (((double)vA_aux[i]-med_A)*((double)vA_aux[i]-med_A));
		E_anm2_B = E_anm2_B	+ (((double)vB_aux[i]-med_B)*((double)vB_aux[i]-med_B));
	}
		
	free(vA_aux);
	free(vB_aux);
	
	div=sqrt(E_anm2_A * E_anm2_B);
	if (div == 0) return undef;
	else return (float)(E_cov/div);
}

float sig_mcarlo_serie_serie(float *vA, float *vB, int nt, float undef, int total_perm){
	
	int i, cont;
	float correl_orig, correl;
	
	correl_orig=correlacao_serie_serie(vA,vB,nt,undef);

	if (correl_orig == undef) return undef;
	
	//srand(time(NULL));
	for (i=0;i<total_perm;i++){
		shuffle(vB,nt);
		correl=correlacao_serie_serie(vA,vB,nt,undef);
		if (fabs(correl) >= fabs(correl_orig)) cont++;
	}
	
	return ((float)cont/(float)total_perm)*(correl_orig*fabs(correl_orig));
}

/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

int main(int argc, char **argv){
	
	int p, pos;

	//LE OS ARGUMENTOS E OS DADOS DE ENTRADA
	le_argumentos(argc,argv,&param);
	le_serie_entrada(param,&h_serie_entrada);
	le_matriz_entrada(param,&h_matriz_entrada);
	
	h_saida=(float*)malloc(param.NP*sizeof(float));
	
	for (p=0; p<param.NP; p++){
		pos=calcula_pos_matriz(param.NT, p, 0);
		h_saida[p]=sig_mcarlo_serie_serie(h_serie_entrada, &(h_matriz_entrada[pos]),param.NT, param.UNDEF, TOTAL_PERM);
	}
	
	salva_arq_saida(param, h_saida);
	
	free(h_matriz_entrada);
	free(h_serie_entrada);
	free(h_saida);
		
	return 1;
}
