#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <time.h>
using namespace std;
#define loop_iter 1000000
const int N=100000;
#pragma omp declare simd
int add(int a, int b)
{
   int c;
   c = a + b ;
   return c;
}
void findAdd( int *a, int *b, int n )
{
   int i;
   int c[N];
   #pragma omp simd
   for ( i = 0; i < n; i++ ) {
      c[i]= add( a[i],  b[i]);
   }
}

int main(){
   int i;
   int a[N], b[N];

   for ( i=0; i<N; i++ ) {
      a[i] = i; b[i] = N-i;
   }

   chrono::steady_clock::time_point begin = chrono::steady_clock::now();
   for ( i=0; i<loop_iter; i++ ) {
      findAdd(a, b, N );
   }
   chrono::steady_clock::time_point end = chrono::steady_clock::now();
  cout <<"Vectorized"<< endl;
  cout <<"Time difference = " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << "[ns]" << endl;
  cout <<"Time difference = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[\xE6s]" << endl;
  cout <<"Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
  cout <<"Time difference = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << "[s]" << endl;

   return 0;
}
