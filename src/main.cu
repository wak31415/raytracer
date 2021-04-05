#include <stdio.h>
#include "cu_vector.h"
#define N 1000

int write_image();

__global__ void vector_add_kernel(CU_Vector3f *c) {
    int i = threadIdx.x;
    if (i < N) {
        CU_Vector3f a({9.f, 6.f, 3.f});

        c[i] = a + 3*c[i];
    }
}

int main() {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CU_Vector3f c[N];
    for(int i=0; i<N; i++) {
        c[i] = CU_Vector3f(1.f, 2.f, 3.f);
    }

    int size = sizeof(CU_Vector<3>) * N;

    CU_Vector3f *d_c;
    cudaMalloc((void**)&d_c, size);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++) {
        printf("Value of c: {%.1f, %.1f, %.1f}\n", c[i][0], c[i][1], c[i][2]);
        for(int j=0; j<3; j++) {
            if (c[i][j] != 12.f) {
                printf("Error: incorrect value (%.1f)\n", c[i][j]);
            }
        }
    }

    printf("Completed operations\n");
    return write_image();
}