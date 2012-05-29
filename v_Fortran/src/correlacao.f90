program correl_bin

	implicit none

	!Numero de permutacoes que serao feitas no teste Monte Carlo
	integer, parameter	:: N_PERM=500

	!Declaracao das funcoes:
	real*8:: correlacao_serie_serie,Sig_Monte_serie_serie

	integer nx,ny,nt
	real*8::undef
	real*4,allocatable::dados_serie(:), dados_bin(:,:,:)
	real*8,allocatable::correl(:,:), signif(:,:)
	character*256 arq_bin,arq_serie,arq_cor,arq_sig

	integer x,y

	call ler_argumentos(arq_serie,arq_bin,nx,ny,nt,undef,arq_cor,arq_sig)

	allocate(dados_bin(nx,ny,nt))
	allocate(dados_serie(nt))
	allocate(correl(nx,ny))
	allocate(signif(nx,ny))

	call ler_serie(arq_serie,dados_serie,nt)
	call ler_binario(arq_bin,dados_bin,nx,ny,nt)

	do x=1,nx
		do y=1,ny
			!write(*,*) 'X: ',x,' / ','Y: ',y
			
			!CALCULA A CORRELACAO
			correl(x,y)=correlacao_serie_serie(dble(dados_serie),dble(dados_bin(x,y,:)),nt,undef)
			
			!CALCULA A SIGNIFICANCIA PELO TESTE DE MONTE CARLO
			!signif(x,y)=Sig_Monte_serie_serie(dble(dados_bin(x,y,1:nt)),dble(dados_serie(1:nt)),nt,N_PERM,undef)
			
			!write(*,'(f10.6,1x,f10.6)') correl(x,y), signif(x,y)

		enddo
	enddo

	call salvar_arquivo(arq_cor,real(correl),nx,ny)
	call salvar_arquivo(arq_sig,real(signif),nx,ny)

end program correl_bin

! -----------------------------------------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------------------------------------

subroutine ler_argumentos(arq_serie,arq_bin,nx,ny,nt,undef,arq_cor,arq_sig)

   implicit none

   integer::nx,ny,nt
   character*256 arg,arq_bin,arq_serie,arq_cor,arq_sig
   real*8::undef
   real*4::und

	if(iargc() .NE. 8)then
		write(*,*)"Sintaxe:"
		write(*,*)"	./correlacao 'arq_serie' 'arq_bin' 'nx' 'ny' 'nt' 'undef' 'arq_cor' 'arq_sig'"
		write(*,*)""
		write(*,*)"Parametros:"
		write(*,*)"	arq_serie: arquivo binario de entrada contendo uma unica serie;"
		write(*,*)"	arq_bin: arquivo binario de entrada contendo multiplas series;"
		write(*,*)"	nx: numero total de colunas da matriz do binario de multiplas series;"
		write(*,*)"	ny: numero total de linhas da matriz do binario de multiplas series;"
		write(*,*)"	nt: numero total de dias da matriz em ambos os arquivos;"
		write(*,*)" undef: valor dos dados indefinidos;"
		write(*,*)"	arq_cor: arquivo de saida, contendo os valores da correlacao;"
		write(*,*)"	arq_sig: arquivo de saida, contendo os valores da significancia da correlacao;"
		write(*,*)""
		write(*,*)"	Exemplo:"
		write(*,*)"	./correlacao Af_1979-1999.bin AmSul_1979-1999.bin 48 56 7665 777.7 cor.bin sig.bin"
		write(*,*)""
		stop
   endif

   call getarg(1,arq_serie)

   call getarg(2,arq_bin)

   call getarg(3,arg)
   read(arg,*) nx

   call getarg(4,arg)
   read(arg,*) ny

   call getarg(5,arg)
   read(arg,*) nt

   call getarg(6,arg)
   read(arg,*) und
   undef=dble(und)

   call getarg(7,arq_cor)

   call getarg(8,arq_sig)

end

! -----------------------------------------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------------------------------------

!Subrotina de leitura do arquivo de entrada.
!Descrição: lê o arquivo de entrada (arq) e armazena os dados na
!           matriz 'dados'.
!Parâmetros: arq: arquivo que será lido;
!            dados: matriz que armazenará os dados do arquivo lido;
!            nx: tamanho da matriz em x;
!            ny: tamanho da matriz em y;
!            nt: tamanho da matriz em t.
subroutine ler_binario(arq,dados,nx,ny,nt)

   implicit none

   integer x,y,t,nx,ny,nt
   real*4::dados(nx,ny,nt)
   character*256 arq

   open(10,file=arq,access='direct',recl=nx*ny*nt*4,status='old')
   read(10,rec=1) (((dados(x,y,t),x=1,nx),y=1,ny),t=1,nt)
   close(10)

end subroutine

! -----------------------------------------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------------------------------------

!Subrotina de leitura do arquivo de entrada.
!Descrição: lê o arquivo de entrada (arq) e armazena os dados na
!           matriz 'dados'.
!Parâmetros: arq: arquivo que será lido;
!            dados: matriz que armazenará os dados do arquivo lido;
!            nt: tamanho da matriz em t.
subroutine ler_serie(arq,dados,nt)

   implicit none

   integer t,nt
   real*4, intent(out)::dados(nt)
   character*256 arq

   open(10,file=arq,access='direct',recl=nt*4,status='old')
   read(10,rec=1) (dados(t),t=1,nt)
   close(10)

end subroutine

! -----------------------------------------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------------------------------------

!Subrotina de armazenamento do arquivo de saída.
!Descrição: armazena os dados da matriz 'dados' no arquivo 'arq'.
!           Caso o arquivo já exista, seu conteúdo será apagado.
!Parâmetros: arq: arquivo onde serão armazenados os dados da matriz 'dados';
!            dados: matriz contendo os dados a serem armazenados em 'arq';
!            nx: tamanho da matriz em x.
!            ny: tamanho da matriz em y.
subroutine salvar_arquivo(arq,dados,nx,ny)

   implicit none

   real::dados(nx,ny)
   integer x,y,nx,ny
   character*256 arq

   open(10,file=trim(arq),access='direct',recl=nx*ny*4,status='replace')
   write(10,rec=1) ((dados(x,y),x=1,nx),y=1,ny)
   close(10)

end subroutine

! -----------------------------------------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------------------------------------

!Funcao correlacao_serie_serie
!
!calcula a correlacao entre duas series
!
!PARAMETRO1: primeiro vetor 	TIPO real*8(fim)
!PARAMETRO2: segundo  vetor 	TIPO real*8(fim)
!PARAMETRO3: indice inicial  	TIPO integer
!PARAMETRO4: indice final	  	TIPO integer
!PARAMETRO4: valor undef	  	TIPO real*8
!
!Modificado em: 02/09/2011, por Bruno N. Zanette:
!- Alterado "real" para "real*8"
!- Adicionado o parametro "nt"

real*8   function correlacao_serie_serie(vetA,vetB,nt,undef)

	implicit  none

	integer, intent(in) :: nt
	real*8,intent (in)  :: vetA(1:nt), vetB(1:nt), undef

	integer  			::  i, k
	real*8				:: auxA(size(vetA)), auxB(size(vetB))
	real*8			    :: med_x, med_y, div, E_cov, E_anm2_x, E_anm2_y

	k = 0
	do i=1,size(vetA)
		auxA(k+1) = vetA(i)
		auxB(k+1) = vetB(i)
		if((auxA(k+1).ne.undef).and.(auxB(k+1).ne.undef))k=k+1
	enddo

	if (k.lt.2)then
		correlacao_serie_serie = undef
		return
	endif

	med_x=0
	med_y=0
	do i=1,k
		med_x=med_x+auxA(i)
		med_y=med_y+auxB(i)
	enddo
	med_x=med_x/float(k)
	med_y=med_y/float(k)
	E_cov=0
	E_anm2_x=0
	E_anm2_y=0

	do i=1,k
		E_cov    = E_cov    + ((auxA(i)-med_x) * (auxB(i)-med_y))
		E_anm2_x = E_anm2_x + ((auxA(i)-med_x) * (auxA(i)-med_x))
		E_anm2_y = E_anm2_y + ((auxB(i)-med_y) * (auxB(i)-med_y))
	enddo
	div=sqrt(E_anm2_x * E_anm2_y)
	if(div.eq.0)then
		correlacao_serie_serie=undef
	else
		correlacao_serie_serie=E_cov/div
	endif

	return

end function correlacao_serie_serie

! -----------------------------------------------------------------------------------------------------------------------
! -----------------------------------------------------------------------------------------------------------------------

!Significância Monte Carlo
!
!Calcula a significancia de duas series usando o teste de Monte Carlos
!
!Modificado em: 02/09/2011, por Bruno N. Zanette:
!- Alterado "real" para "real*8"
!- Adicionado o parametro "nt"

real*8 function Sig_Monte_serie_serie(array1,array2,nt,permutations,undef)

	USE MODRANDOM

	implicit  none

	!Declaracoes das funcoes usadas:
	real*8	:: correlacao_serie_serie

	integer, intent(in) :: nt
	real*8,intent(in)	:: array1(1:nt),array2(1:nt),undef
	integer,intent(in)	:: permutations

	integer				:: i,count
	real*8				:: array1_(1:nt),array2_(1:nt)

	real*8				:: correl_orig
	real*8				:: correl

	correl_orig=correlacao_serie_serie(array1,array2,nt,undef)
	array1_=array1
	array2_=array2

	call set_seeds(1)
	if (correl_orig /= undef) then
		count=0;
		do i=1,permutations
			call random_permutation(array1_,size(array1_))
!			call random_permutation(array2_,size(array2_))
			correl=correlacao_serie_serie(array1_,array2_,nt,undef)
			if (abs(correl) >= abs(correl_orig) ) then
				count=count+1		
			end if 
		end do
	    Sig_Monte_serie_serie=1.-real(count)/real(permutations)
	    Sig_Monte_serie_serie=Sig_Monte_serie_serie*(correl_orig/abs(correl_orig))
		!write(*,*)count,permutations,Sig_Monte_serie_serie
		return 
	else
	    Sig_Monte_serie_serie=undef
		return
	end if

end function Sig_Monte_serie_serie
