/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"

#pragma omp declare simd linear(k:1)
float vect1(float c0,float c1, float *A0,const int nx, const int ny,int i, int j, int k){
	return (A0[Index3D (nx, ny, i, j, k + 1)] +
				A0[Index3D (nx, ny, i, j, k - 1)] +
				A0[Index3D (nx, ny, i, j + 1, k)] +
				A0[Index3D (nx, ny, i, j - 1, k)] +
				A0[Index3D (nx, ny, i + 1, j, k)] +
				A0[Index3D (nx, ny, i - 1, j, k)])*c1
				- A0[Index3D (nx, ny, i, j, k)]*c0;
}

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{

  int i, j, k;
  #pragma omp simd collapse(3)
	for(i=1;i<nx-1;i++)
	{
		for(j=1;j<ny-1;j++)
		{
			for(k=1;k<nz-1;k++)
			{
				Anext[Index3D (nx, ny, i, j, k)] = vect1(c0,c1,A0,nx,ny,i,j,k);
			}
		}
	}

}


