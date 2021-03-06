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
/* Default data type is double, default size is 10x1024x1024. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(X,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	X[i][j] = ((DATA_TYPE) i*(j+1) + 1) / n;
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(X,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
    static
void kernel_adi(int tsteps,
        int n,
        DATA_TYPE POLYBENCH_2D(X,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{

    #pragma omp target data map(tofrom: X[0:n]) map(to: A[0:n], B[0:n]) //acc data copy(X) copyin(A,B)
    {
        //#pragma acc parallel
        {
            for (int t = 0; t < _PB_TSTEPS; t++)
            {
                #pragma omp target teams distribute parallel for schedule(static,1) \
                    collapse(1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS) 
                for (int i1 = 0; i1 < N; i1++)
                    for (int i2 = 1; i2 < N; i2++)
                    {
                        X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
                        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
                    }

                //#pragma omp target teams distribute parallel for schedule(static,1) \
                    collapse(1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS)
                for (int i1 = 0; i1 < N; i1++)
                    X[i1][N-1] = X[i1][N-2] / B[i1][N-1];

                #pragma omp target teams distribute parallel for schedule(static,1) \
                    collapse(1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS)
                for (int i1 = 0; i1 < N; i1++)
                    for (int i2 = 0; i2 < N-2; i2++)
                        X[i1][N-i2-2] = (X[i1][N-2-i2] - X[i1][N-2-i2-1] * A[i1][N-i2-3]) / B[i1][N-3-i2];

                #pragma omp target teams distribute parallel for schedule(static,1) \
                    collapse(1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS)
                for (int i1 = 1; i1 < N; i1++)
                    for (int i2 = 0; i2 < N; i2++) {
                        X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
                        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
                    }

                //#pragma omp target teams distribute parallel for schedule(static,1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS)
                for (int i2 = 0; i2 < N; i2++)
                    X[N-1][i2] = X[N-1][i2] / B[N-1][i2];

                #pragma omp target teams distribute parallel for schedule(static,1) \
                    collapse(1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS)
                for (int i1 = 0; i1 < N-2; i1++)
                    for (int i2 = 0; i2 < N; i2++)
                        X[N-2-i1][i2] = (X[N-2-i1][i2] - X[N-i1-3][i2] * A[N-3-i1][i2]) / B[N-2-i1][i2];

                
                // XXX DEBUG
                //#pragma omp target teams distribute parallel for schedule(static,1) \
                    collapse(1) \
                    num_teams(NUM_TEAMS) \
                    num_threads(NUM_THREADS)
                //for (int i1 = 0; i1 < N-2; i1++)
                //    for (int i2 = 0; i2 < N; i2++) {
                //        X[N-2-i1][i2] = X[N-2-i1][i2] - X[N-i1-3][i2];
                //    }

                //#pragma omp target teams distribute parallel for schedule(static,1) \
                //    collapse(1) \
                //    num_teams(NUM_TEAMS) \
                //    num_threads(NUM_THREADS)
                //for (int i1 = 0; i1 < N-2; i1++)
                //    for (int i2 = 1; i2 < N; i2++)
                //        X[N-2-i1][i2] = X[N-3-i1][i2-1];
                
                //#pragma omp target teams distribute parallel for schedule(static,1) \
                //    collapse(1) \
                //    num_teams(NUM_TEAMS) \
                //    num_threads(NUM_THREADS)
                //for (int i1 = 0; i1 < N-2; i1++)
                //    for (int i2 = 0; i2 < N; i2++)
                //        X[i1 + 2][i2] = 1.0;
                // XXX END DEBUG
                
            }
        }
    }

    printf("X[1][1] = %f\n", X[1][1]);
    //printf("X[1000][1000] = %f\n", X[1000][1000]);
    printf("B[1][1] = %f\n", B[1][1]);
    printf("B[1000][1000] = %f\n", B[1000][1000]);
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(X, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(X),
	      POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(X);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
