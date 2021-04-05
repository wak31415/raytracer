#include <cuda.h>

template <size_t MatSize>
class CU_Matrix {
    public:
        __host__ __device__ CU_Matrix() : N(MatSize) {
            memset(data, 0, N*N*sizeof(float));
        }

        __host__ __device__ CU_Matrix(float *data) : N(MatSize) {
            memcpy(this->data, data, N*N*sizeof(float));
        }

        __host__ __device__ size_t get_size() { return N; }

        __host__ __device__ const float& operator[](int i) const { return data[i]; }
        __host__ __device__ float& operator[](int i) { return data[i]; }

    protected:
        int N;
        float data[MatSize*MatSize];
};

template <size_t VecSize>
__host__ __device__ CU_Vector<VecSize> operator*(const CU_Matrix<VecSize> &M, const CU_Vector<VecSize> &v) {
    CU_Vector<VecSize> R;

    for (size_t i = 0; i < VecSize; i++)
    {
        float tmp = 0.f;
        for (size_t j = 0; j < VecSize; j++)
            tmp += M[i*VecSize + j] * v[j];

        R[i] = tmp;
    }
    return R;
}

template <size_t MatSize>
__host__ __device__ CU_Matrix<MatSize> operator*(const CU_Matrix<MatSize> &A, const CU_Matrix<MatSize> &B) {
    CU_Matrix<MatSize> C, B_transpose;

    // Transpose other matrix first to improve sequential memory access
    for (size_t i = 0; i < MatSize; i++) {
        for (size_t j = 0; j < MatSize; j++) {
            B_transpose[i+j*MatSize] = B[i*MatSize + j];
        }
    }

    // Actual matrix multiplication
    for (size_t i = 0; i < MatSize; i++) {
        for (size_t j = 0; j < MatSize; j++) {
            float tmp = 0.f;
            for (size_t k = 0; k < MatSize; k++) {
                tmp += A[i*MatSize + k] * B_transpose[j*MatSize + k];
            }
            C[i*MatSize + j] = tmp;
        }
    }
    return C;
}

