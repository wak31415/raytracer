template <size_t MatSize>
class CU_Matrix {
    public:
        __host__ __device__ CU_Matrix() {
            memset(data, 0, MatSize*MatSize*sizeof(float));
        }

        __host__ __device__ CU_Matrix(float *data) {
            memcpy(this->data, data, MatSize*MatSize*sizeof(float));
        }

        __host__ __device__ void set_identity() {
            memset(data, 0, MatSize*MatSize*sizeof(float));
            for(size_t i = 0; i < MatSize; i++) {
                data[i*MatSize + i] = 1.f;
            }
        }

        __host__ __device__ CU_Matrix<3> get_rotation() {
            if(MatSize != 4) return CU_Matrix<3>();

            CU_Matrix<3> res;
            for(size_t i = 0; i < 3; i++) {
                for(size_t j = 0; j < 3; j++) {
                    res(i, j) = data[i*MatSize + j];
                }
            }
            return res;
        }

        __host__ __device__ CU_Vector3f get_translation() {
            if(MatSize != 4) return CU_Vector3f();

            CU_Vector3f res;
            for(size_t i = 0; i < 3; i++) {
                res[i] = data[i*MatSize + 3];
            }
            return res;
        }

        __host__ __device__ size_t get_size() { return MatSize; }

        __host__ __device__ const float& operator[](int i) const { return data[i]; }
        __host__ __device__ float& operator[](int i) { return data[i]; }
        __host__ __device__ float& operator()(int i, int j) { return data[i*MatSize + j]; }

    protected:
        float data[MatSize*MatSize];
};

template <size_t VecSize, typename T>
__host__ __device__ CU_Vector<VecSize, float> operator*(const CU_Matrix<VecSize> &M, const CU_Vector<VecSize, T> &v) {
    CU_Vector<VecSize, float> R;

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

