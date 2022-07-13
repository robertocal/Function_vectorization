/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359
float kernel_value_CPU(float v){

  float rValue = 0;

  const float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
  (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
   0.479440257548300e-16f) + 0.435125971262668e-13f ) +
   0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
   0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
   0.463076284721000e0f)   + 0.754337328948189e2f   ) +
   0.830792541809429e4f)   + 0.571661130563785e6f   ) +
   0.216415572361227e8f)   + 0.356644482244025e9f   ) +
   0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  rValue = -num/den;

  return rValue;
}
#pragma omp declare simd inbranch uniform(sdc)
float kernel_value_CPU_vect(float v, float sdc){
  //v = beta*sqrt(1.0-(v*_1overCutoff2));

  float rValue = 0;

  const float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
  (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
   0.479440257548300e-16f) + 0.435125971262668e-13f ) +
   0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
   0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
   0.463076284721000e0f)   + 0.754337328948189e2f   ) +
   0.830792541809429e4f)   + 0.571661130563785e6f   ) +
   0.216415572361227e8f)   + 0.356644482244025e9f   ) +
   0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  rValue = -num/den;

  return rValue*sdc;
}
#pragma omp declare simd inbranch
float kernel_value_CPU_vect_1(float v){
    const float z = v*v;
    float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
    (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
    0.479440257548300e-16f) + 0.435125971262668e-13f ) +
    0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
    0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
    0.463076284721000e0f)   + 0.754337328948189e2f   ) +
    0.830792541809429e4f)   + 0.571661130563785e6f   ) +
    0.216415572361227e8f)   + 0.356644482244025e9f   ) +
    0.144048298227235e10f);
    return num;
}
#pragma omp declare simd inbranch
float kernel_value_CPU_vect_2(float v){
  const float z = v*v;
  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);
  return den;
}
#pragma omp declare simd inbranch
float kernel_value_CPU_vect_3(float num, float den, float sdc){
  float rValue = 0;
  return rValue = (-num/den)*sdc;
}
void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  float v;
  float cutoff2 = (width*width)/4.0;

  unsigned int size;

  if(width > 0){
    // compute size of LUT based on kernel width
    size = (unsigned int)(10000*width);

    // allocate memory
    (*LUT) = (float*) malloc (size*sizeof(float));

    unsigned int k;
    for(k=0; k<size; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      v = (((float)k)/((float)size))*cutoff2;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrt(1.0-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}
#pragma omp declare simd inbranch uniform(sizeLUT,_1overCutoff2, sdc)
float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2,float sdc)
{
  unsigned int k0;
  float v0;
  float sub;
  float sub2;
  float div;

  float mul;
  v *= (float)sizeLUT;
  k0=(unsigned int)(v*_1overCutoff2);
  v0 = ((float)k0)/_1overCutoff2;
  sub = LUT[k0+1]-LUT[k0];
  sub2 = v-v0;
  mul = sub2*sub;
  div = mul/_1overCutoff2;
  return (LUT[k0] + div)*sdc;
}
#pragma omp declare simd inbranch uniform(sizeLUT)
float kernel_value_LUT_Vect(float v, int sizeLUT){
  return v * (float)sizeLUT;
}
#pragma omp declare simd inbranch
unsigned int kernel_value_LUT_Vect_2(float v, float _1overCutoff2){
  return v*_1overCutoff2;
}
#pragma omp declare simd inbranch uniform(_1overCutoff2)
float kernel_value_LUT_Vect_3(int k0, float _1overCutoff2){
  return ((float)k0)/_1overCutoff2;
}
#pragma omp declare simd inbranch
float kernel_value_LUT_Vect_4(float a, float b){
  return a - b;
}
#pragma omp declare simd inbranch
float kernel_value_LUT_Vect_5(float sub2, float sub){
  return sub2*sub;
}
#pragma omp declare simd inbranch
float kernel_value_LUT_Vect_6(float mul, float _1overCutoff2){
  return mul/_1overCutoff2;
}
#pragma omp declare simd inbranch
float kernel_value_LUT_Vect_7(float div, float b){
  return div + b;
}
#pragma omp declare simd inbranch
float kernel_value_LUT_Vect_8(float fin, float sdc){
  return fin*sdc;
}
#pragma omp declare simd inbranch uniform(x) linear(y:1)
float loop_1(float x, int y){
  return ((x-y)*(x-y));
}
#pragma omp declare simd inbranch linear(k:1)
float loop_2(float dy2dz2,float *dx2, int k){
  return dy2dz2+(dx2[k]);
}
#pragma omp declare simd linear(k:1)
float loop_3(float dy2dz2,float *dx2, int k){
  return dy2dz2+(dx2[k]);
}
int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  unsigned int NzL, NzH;

  int nx;
  int ny;
  int nz;

  float w;
  unsigned int idx;
  unsigned int idx0;

  unsigned int idxZ;
  unsigned int idxY;

  float Dx2[100];
  float Dy2[100];
  float Dz2[100];
  float *dx2=NULL;
  float *dy2=NULL;
  float *dz2=NULL;
  float *dz22=NULL;
  float dy2dz2;
  float v;

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))/2.0; // cutoff radius
  float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  float _1overCutoff2 = 1/cutoff2;                  // 1 over square of cutoff radius

  float beta = PI * sqrt(4*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-.5)*(params.oversample-.5)-.8);

  int i;
  int j;
  int h;
  int k;

  printf("%d\n",Dz2);
  printf("%d\n",&Dz2[0]);
  /*dz2 = Dz2;
  dz22 = Dz2;
  *dz2 =6;
  dz2[0] = 6;
  ++dz2;
  printf("%d\n",dz2);
  printf("%d\n",&dz2[0]);
  printf("%d\n",&dz22[1]);//=&dz2[0]
  printf("%d\n",&dz22[2]);//=&dz2[1]
  printf("%d\n",&dz2[-1]);
  printf("%d\n",&dz2[1]);
  printf("%d\n",&Dz2[1]);
  dz2[1] = 12;
  //printf("%f\n",dz2[1]);
  return 0;*/
  for (i=0; i < n; i++){
    ReconstructionSample pt = sample[i];

    float kx = pt.kX;
    float ky = pt.kY;
    float kz = pt.kZ;

    NxL = max((kx - cutoff), 0.0);
    NxH = min((kx + cutoff), size_x-1.0);

    NyL = max((ky - cutoff), 0.0);
    NyH = min((ky + cutoff), size_y-1.0);

    NzL = max((kz - cutoff), 0.0);
    NzH = min((kz + cutoff), size_z-1.0);

    if((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc!=0.0)
    {
      j = 0;
      dz2 = Dz2;
      #pragma omp simd simdlen(6)
      for(nz=NzL; nz<=NzH; ++nz)
      {
        ++j;
        dz2[j] = loop_1(kz, nz);
        //printf("%f\n",dz2[j]);
      }
      dx2=Dx2;
      j = 0;
      #pragma omp simd simdlen(6)
      for(nx=NxL; nx<=NxH; ++nx)
      {
        ++j;
        dx2[j] = loop_1(kx, nx);
      }
      j = 0;
      dy2=Dy2;
      #pragma omp simd simdlen(6)
      for(ny=NyL; ny<=NyH; ++ny)
      {
        ++j;
        dy2[j] = loop_1(ky, ny);
      }
      idxZ = (NzL-1)*size_x*size_y;
      j = 0;
      for(dz2=Dz2, nz=NzL; nz<=NzH; ++nz)
      {
        ++j;
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += size_x*size_y;

        idxY = (NyL-1)*size_x;

        /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding x&y distance) */
        if((dz2[j])<cutoff2)
        {
          h = 0;
          for(dy2=Dy2, ny=NyL; ny<=NyH; ++ny)
          {
            ++h;
            /* linear offset IN ADDITION to idxZ to get to Y position */
            idxY += size_x;

            dy2dz2=(dz2[j])+(dy2[h]);

            idx0 = idxY + idxZ;

            /* loop over y indexes, but only if curent distance is close enough (distance will increase by adding y distance) */
            if(dy2dz2<cutoff2)
            {
              k = 0;
              dx2=Dx2;
              #pragma omp simd private(v,idx, w)
              for(nx=NxL; nx<=NxH; ++nx){
                ++k;
                /* value to evaluate kernel at */
                //v = dy2dz2+(dx2[k]);
                v = loop_2(dy2dz2,dx2,k);
                if(v<cutoff2)
                {
                  /* linear index of (x,y,z) point */
                  idx = nx + idx0;

                  /* kernel weighting value */
                  if (params.useLUT){
                    float v_vect = kernel_value_LUT_Vect(v, sizeLUT);
                    unsigned int k0 = kernel_value_LUT_Vect_2(v_vect, _1overCutoff2);
                    float v0 = kernel_value_LUT_Vect_3(k0, _1overCutoff2);
                    float sub1 = kernel_value_LUT_Vect_4(LUT[k0+1], LUT[k0]);
                    float sub2 = kernel_value_LUT_Vect_4(v_vect, v0);
                    float mul = kernel_value_LUT_Vect_5(sub2, sub1);
                    float div = kernel_value_LUT_Vect_6(mul, _1overCutoff2);
                    float fin = kernel_value_LUT_Vect_7(div, LUT[k0]);
                    w = kernel_value_LUT_Vect_8(fin,pt.sdc);
                    //w = kernel_value_LUT(v,LUT,sizeLUT,_1overCutoff2,pt.sdc);
                  } else {
                    float in = beta*sqrt(1.0-(v*_1overCutoff2));
                    float num = kernel_value_CPU_vect_1(in);
                    float den = kernel_value_CPU_vect_2(in);
                    w = kernel_value_CPU_vect_3(num, den,pt.sdc);
                    //w = kernel_value_CPU(beta*sqrt(1.0-(v*_1overCutoff2))) * pt.sdc;
                  }

                  /* grid data */
                  gridData[idx].real += (w*pt.real);
                  gridData[idx].imag += (w*pt.imag);

                  /* estimate sample density */
                  sampleDensity[idx] += 1.0;
                }
              }
            }
          }
        }
      }
    }
  }
}
