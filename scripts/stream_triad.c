/* Minimal STREAM-style triad, a[i] = b[i] + s*c[i], to measure sustainable DRAM
 * bandwidth for the roofline comparison in the report. Arrays are far larger
 * than the L3 cache. Bandwidth counts 24 bytes per element (two reads, one
 * write), as STREAM does; write-allocate traffic is not counted.
 *   cc -O3 -march=native -fopenmp scripts/stream_triad.c -o stream_triad
 *   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close ./stream_triad
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const size_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : (size_t)1 << 25; /* 256 MiB per array */
    const int trials = 10;
    double *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b), *c = malloc(n * sizeof *c);
    if (!a || !b || !c) { fprintf(stderr, "allocation failed\n"); return 1; }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; } /* first touch per thread */
    double best = 1e30, scalar = 3.0;
    for (int t = 0; t < trials; ++t) {
        double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) a[i] = b[i] + scalar * c[i];
        double elapsed = omp_get_wtime() - start;
        if (t > 0 && elapsed < best) best = elapsed; /* skip the first trial */
    }
    double check = 0.0;
    for (size_t i = 0; i < n; i += n / 16) check += a[i];
    printf("threads,array_mib,best_s,triad_gb_s,check\n%d,%.0f,%.6f,%.2f,%.1f\n", omp_get_max_threads(),
           n * sizeof(double) / 1048576.0, best, 24.0 * n / best / 1e9, check);
    free(a); free(b); free(c);
    return 0;
}
