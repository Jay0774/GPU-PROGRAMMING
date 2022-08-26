#include<stdio.h>
#include<stdlib.h>
#include<immintrin.h>
#include<omp.h>
#include<time.h>
int n;

void add_parallel(double *a, double *b, double *c)
{
        int i;
        #pragma parallel for
        for(i=0;i<n;i++)
        {
                c[i]=a[i]+b[i];
        }
}

void add_sequential(double *a, double *b, double *c)
{
        int i;
        for(i=0;i<n;i++)
        {
                c[i]=a[i]+b[i];
        }
}

void add_vector(double *a,double *b,double *c)
{
        int i;
        for(i=0;i<n;i+=8)
        {
                const  __m512d t1 = _mm512_load_pd(&a[i]);
                const  __m512d t2 = _mm512_load_pd(&b[i]);
                const  __m512d res = _mm512_add_pd(t1,t2);
                _mm512_stream_pd(&c[i],res);

        }
}

int main(int argc,char **argv)
{
        omp_set_num_threads(20);
        if(argc==2)
        {
                n = atoi(argv[1]);
                double *a = (double *)aligned_alloc(1024,n*sizeof(double));
                double *b = (double *)aligned_alloc(1024,n*sizeof(double));
                double *c = (double *)aligned_alloc(1024,n*sizeof(double));
                int i;
                clock_t start,stop;
                for(i=0;i<n;i++)
                {
                        a[i] = 1.11;
                        b[i] = 2.22;
                }
                start = clock();
                add_sequential(a,b,c);
                stop = clock();
                printf("Time taken by sequetial add is    :  %lf ms\n",(double)(stop-start)*10e-3);

                start = clock();
                add_vector(a,b,c);
                stop = clock();
                printf("Time taken by vectorised add is   :  %lf ms\n",(double)(stop-start)*10e-3);

                start = clock();
                add_parallel(a,b,c);
                stop = clock();
                printf("Time taken by parallel add is     :  %lf ms\n",(double)(stop-start)*10e-3);
        }
        return 0;
}
