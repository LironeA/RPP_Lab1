#include <omp.h>
#include <stdlib.h>
#include <float.h>

void floyd_warshall_omp(double* dist, int n) {
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dik = dist[i * n + k];
                double dkj = dist[k * n + j];
                double dij = dist[i * n + j];
                if (dik + dkj < dij) {
                    dist[i * n + j] = dik + dkj;
                }
            }
        }
    }
}