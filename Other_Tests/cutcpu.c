/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "atom.h"
#include "cutoff.h"

#undef DEBUG_PASS_RATE
#define CHECK_CYLINDER_CPU

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

#pragma omp declare simd uniform(x) linear(i:1) notinbranch
void loop1(int *x, int i){
  x[i] = -1;
}
/*#pragma omp declare simd inbranch
void loop2(float r2, float s, float inv_a2, float *pg, float e, float q){
  s = (1.f - r2 * inv_a2);
  e = q * (1/sqrtf(r2)) * s * s;
  *pg += e;
}*/
#pragma omp declare simd notinbranch
float loop2(float dx, float dydz2){
  return dx*dx + dydz2;
}
#pragma omp declare simd notinbranch
float loop3(float r2, float inv_a2){
  return (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
}
#pragma omp declare simd notinbranch
float loop4(float r2, float s, float q){
  return q * (1/sqrtf(r2)) * s;
}
#pragma omp declare simd inbranch
void loop5(float e, float *pg){
  *pg += e;
}
#pragma omp declare simd
float loop6(float dx, float gridspacing){
  return dx += gridspacing;
}
extern int cpu_compute_cutoff_potential_lattice(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms                       /* array of atoms */)
{
  printf("a\n");
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float gridspacing = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  const float a2 = cutoff * cutoff;
  const float inv_a2 = 1.f / a2;
  float s;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
    /* lattice point radius about each atom */

  int n;
  int i, j, k;
  int ia, ib, ic;
  int ja, jb, jc;
  int ka, kb, kc;
  int index;
  int koff, jkoff;

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float e;
  float xstart, ystart;

  float *pg;
  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;		/* Extent of atom bounding box */
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;

#if DEBUG_PASS_RATE
  unsigned long long pass_count = 0;
  unsigned long long fail_count = 0;
#endif

  /* find min and max extent */
  get_atom_extent(&minext, &maxext, atoms);

  /* number of cells in each dimension */
  nxcell = (int) floorf((maxext.x-minext.x) * inv_cellen) + 1;
  nycell = (int) floorf((maxext.y-minext.y) * inv_cellen) + 1;
  nzcell = (int) floorf((maxext.z-minext.z) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  /* allocate for cursor link list implementation */
  first = (int *) malloc(ncell * sizeof(int));
  #pragma omp simd
  for (gindex = 0;  gindex < ncell;  gindex++) {
    loop1(first, gindex);
  }
  next = (int *) malloc(natoms * sizeof(int));
  #pragma omp simd
  for (n = 0;  n < natoms;  n++) {
    loop1(next, n);
  }
  /* geometric hashing */
  for (n = 0;  n < natoms;  n++) {//dopo
    if (0!=atom[n].q){  /* skip any non-contributing atoms */
      i = (int) floorf((atom[n].x - minext.x) * inv_cellen);
      j = (int) floorf((atom[n].y - minext.y) * inv_cellen);
      k = (int) floorf((atom[n].z - minext.z) * inv_cellen);
      gindex = (k*nycell + j)*nxcell + i;
      next[n] = first[gindex];
      first[gindex] = n;
    }
  }

  /* traverse the grid cells */
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      /* find closest grid point with position less than or equal to atom */
      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

      /* find extent of surrounding box of grid points */
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      /* trim box edges so that they are within grid point lattice */
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

      /* loop over surrounding grid points */
      xstart = ia*gridspacing - x;
      ystart = ja*gridspacing - y;
      dz = ka*gridspacing - z;
      for (k = ka;  k <= kb;  k++, dz += gridspacing) {
        koff = k*ny;
        dz2 = dz*dz;
        dy = ystart;
        for (j = ja;  j <= jb;  j++, dy += gridspacing) {
          jkoff = (koff + j)*nx;
          dydz2 = dy*dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx = xstart;
          index = jkoff + ia;
          pg = lattice->lattice + index;
#if defined(__INTEL_COMPILER)
          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            s = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
            e = q * (1/sqrtf(r2)) * s;
            *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! *///vedere questo
          }
#else     
          printf("prima %f\n",dx);
          #pragma omp simd lastprivate(r2,s,e) reduction(+:dx)
          for (i = ia;  i <= ib;  i++) {
            r2 = loop2(dx, dydz2);
            s = loop3(r2, inv_a2);
            e = loop4(r2, s, q);
            printf("dopo %f\n",dx);
            //*pg += (r2 < a2 ? e : 0);
            if(r2<a2){
              *pg += e;
            }
            else{
              *pg += 0;
            }
            //loop6(pg,&dx,gridspacing);
            pg++;
            dx += gridspacing;
          }
          printf("dopo %f\n",dx);
        #endif
        }
      } /* end loop over surrounding grid points */

    } /* end loop over atoms in a gridcell */
  } /* end loop over gridcells */
  /* free memory */
  printf("dopo %f\n",r2);
  printf("dopo %f\n",s);
  printf("dopo %f\n",e);

  free(next);
  free(first);

  /* For debugging: print the number of times that the test passed/failed */
#ifdef DEBUG_PASS_RATE
  printf ("Pass :%lld\n", pass_count);
  printf ("Fail :%lld\n", fail_count);
#endif

  return 0;
}
