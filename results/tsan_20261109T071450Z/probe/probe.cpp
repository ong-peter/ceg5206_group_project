#include <omp.h>
int main() { int x[64] = {};
#pragma omp parallel for num_threads(2)
for (int i = 0; i < 64; ++i) x[i] = i;
return x[63] == 63 ? 0 : 1; }
