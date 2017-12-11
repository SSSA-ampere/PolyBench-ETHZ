/* POLYBENCH/GPU-OPENACC
 *
 * This file is a part of the Polybench/GPU-OpenACC suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE) i*j) / ni;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}

/*
#pragma omp declare target
void __prem_init(void * channel) {

}

void __prem_fini() {

}

void __prem_notify(int pid, int Miid, int miid) {

}
#pragma omp end declare target
*/

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  //#pragma scop
  //#pragma acc data copyin(A,B) copy(C)
  #pragma omp target data map(tofrom: C[0:NI]) map(to: A[0:NI], B[0:NK])
  {
    //#pragma acc parallel
    {
      /* C := alpha*A*B + beta*C */
      //#pragma acc loop
      #pragma omp target teams distribute parallel for schedule(static, 1) \
        num_teams(NUM_TEAMS) \
        num_threads(NUM_THREADS)
        for (int j = 0; j < NJ; j++)
            for (int i = 0; i < NI; i++)
            {
                //C[i][j] *= beta;
                for (int k = 0; k < NK; ++k) {
                    C[i][j] += alpha * A[i][k] * B[k][j] 
#if 0
                        * alpha // 1
                        * alpha // 2
                        * alpha
                        * alpha // 4
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 8
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 16
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 32
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha //64
                        * alpha // 1 (2)
                        * alpha // 2 (2)
                        * alpha
                        * alpha // 4 (2)
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 8 (2)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 16 (2)   // = 80
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 32 (2)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha 
                        * alpha // 64 * 2 = 128
                        * alpha // 1 (3)
                        * alpha // 2 (3)
                        * alpha
                        * alpha // 4 (3)
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 8 (3)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 16 (3)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 32 (3)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha 
                        * alpha // 192
                        * alpha // 1 (4)
                        * alpha // 2 (4)
                        * alpha
                        * alpha // 4 (4)
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 8 (4)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 16 (4)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 224
                        * alpha // 1 (5)
                        * alpha // 2 (5)
                        * alpha
                        * alpha // 4 (5)
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 8 (5)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 16 (5)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 256
                        * alpha // 1 (6)
                        * alpha // 2 (6)
                        * alpha
                        * alpha // 4 (6)
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 8 (6)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 16 (6)
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha
                        * alpha // 288
#endif
                        ;
                }
            }
    }
  }
  //#pragma endscop

  printf("C[4][2] = %f\n", C[4][2]);
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
