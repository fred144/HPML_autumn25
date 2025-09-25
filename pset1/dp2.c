/*copy pasted just same as dp1 but dpunroll*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timing.h"

/// @brief Dot product of two vectors
/// @param N  vector size
/// @param pA pointer to vector A
/// @param pB pointer to vector B
/// @return dot product of A and B
/// @note './dp1 1000 10’ performs 10 measurements on a dot product with vectors of size 1000
float dpunroll(long N, float *pA, float *pB)
{
    float R = 0.0;
    int j;
    for (j = 0; j < N; j += 4)
        R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] + pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3];
    return R;
}
int main(int argc, char **argv)
{

    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s N Reps\n", argv[0]);
        return 1;
    }
    long N = atol(argv[1]);    /*vector size*/
    long Reps = atol(argv[2]); /*number of repetitions*/

    double bytes = (double)N * 2.0 * sizeof(float); // 2 vectors, read only
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);

    float *A = aligned_alloc(64, sizeof(float) * N);
    float *B = aligned_alloc(64, sizeof(float) * N);

    for (long i = 0; i < N; i++) /*initialize with 1.0 */
    {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    double *times = malloc(sizeof(double) * Reps);

    volatile float dot_product; /*per Ed discussion*/

    for (int r = 0; r < Reps; r++)
    {
        double t0 = now_sec();
        dot_product = dpunroll(N, A, B);
        double t1 = now_sec();
        times[r] = t1 - t0;
        // printf("%f\n", dot_product);
        printf("run %d: %f sec, result=%f\n", r, times[r], dot_product);
    }

    int start = Reps / 2; // discard first half
    int count = Reps - start;

    // arithmetic mean of times
    double sumT = 0.0;
    for (int r = start; r < Reps; r++)
        sumT += times[r];
    double meanT = sumT / count;

    // harmonic mean of bandwidth (GB/s) and FLOPS
    double sumInvB = 0.0, sumInvF = 0.0;
    for (int r = start; r < Reps; r++)
    {
        double b = gb / times[r];
        double f = (2.0 * N) / times[r] / 1e9;
        sumInvB += 1.0 / b;
        sumInvF += 1.0 / f;
    }
    double harmB = count / sumInvB;
    double harmF = count / sumInvF;

    printf("\nN: %ld <T>: %f sec  <B>: %f GB/sec  <F>: %f GFLOP/sec\n",
           N, meanT, harmB, harmF);

    free(A);
    free(B);
    free(times);
    return 0;
}