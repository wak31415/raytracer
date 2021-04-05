#include <stdio.h>
#include "cu_vector.h"
#include "cu_matrix.h"
#define N 1000

int write_image();

__global__ void vector_add_kernel(CU_Vector3f *c) {
    int i = threadIdx.x;
    if (i < N) {
        float I_data[9] = {1.f, 0.f, 0.f, 
                           0.f, 1.f, 0.f, 
                           0.f, 0.f, 1.f};

        float M_data[9] = {4.f, 0.f, 0.f, 
                           0.f, 2.f, 0.f, 
                           0.f, 0.f, 1.f};

        CU_Matrix<3> Id(I_data);
        CU_Matrix<3> M(M_data);

        c[i] = Id*M*c[i];
    }
}

int main() {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CU_Vector3f c[N];
    for(int i=0; i<N; i++) {
        c[i] = CU_Vector3f(1.f, 2.f, 4.f);
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
            if (c[i][j] != 4.f) {
                printf("Error: incorrect value (%.1f)\n", c[i][j]);
            }
        }
    }

    printf("Completed operations\n");
    return write_image();
}