/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
//aggiungere questo
#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#pragma omp declare simd
float vect1(float real, float imag){
  return real*real + imag*imag;
}
#pragma omp declare simd
float vect2(struct kValues *kVals, int indexK, float *x, int indexX, float *y, float *z){
  return PIx2 * (kVals[indexK].Kx * x[indexX] + kVals[indexK].Ky * y[indexX] + kVals[indexK].Kz * z[indexX]);
}
#pragma omp declare simd
float vect3(float expArg){
  return cosf(expArg);
}
#pragma omp declare simd
float vect4(float expArg){
  return sinf(expArg);
}
#pragma omp declare simd
float vect6(float phi, float cosArg, float q){
  return (phi*cosArg)+q;
}

//inline 
void ComputePhiMagCPU(int numK, float* phiR, float* phiI, float* phiMag) {
  #pragma omp simd
  for (int indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = vect1(real, imag);
  }
}

//inline
void ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;
  for (indexK = 0; indexK < numK; indexK++) {
    #pragma omp simd private(expArg,cosArg,sinArg)
    for (indexX = 0; indexX < numX; indexX++) {
      expArg = vect2(kVals, indexK, x, indexX, y, z);
      cosArg = vect3(expArg);
      sinArg = vect4(expArg);
      float phi = kVals[indexK].PhiMag;
      Qr[indexX] = vect6(phi, cosArg,Qr[indexX]);
      Qi[indexX] = vect6(phi, sinArg,Qi[indexX]);
    }
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
