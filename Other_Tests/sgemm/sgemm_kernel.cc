/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Base C implementation of MM
 */


#include<iostream>
#pragma declare simd
float vect(float c, const float* A, const float* B, int mm, int i, int lda, int nn, int ldb){
  float a = A[mm + i * lda]; 
  float b = B[nn + i * ldb];
  return c += a * b;
}
#pragma declare simd
float vect2(float* C, float beta, float alpha, float c, int mm, int nn, int ldc){
  return C[mm+nn*ldc] * beta + alpha * c;
}
void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  float c;
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      c = 0.0f;
      #pragma omp simd
      for (int i = 0; i < k; ++i) {
        c = vect(c,A,B,mm,i,lda,nn,ldb);
        //printf("%f\n",c);
      }
      printf("%f\n",c);
      C[mm+nn*ldc] = vect2(C,beta,alpha,c,mm,nn,ldc);
    }
  }
}
