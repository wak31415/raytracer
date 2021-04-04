#include <cuda.h>

template <int VecSize>
class CU_Vector {
    public:
        __host__ __device__ CU_Vector() : N(VecSize) {
            memset(data, 0, sizeof(data));
        };

        __host__ __device__ CU_Vector(float* data, int N) : N(VecSize) {
            this->data = data;
        }

        __host__ __device__ int get_size() { return N; }

        __host__ __device__ const float& operator[](int i) const { return data[i]; }
        __host__ __device__ float& operator[](int i) { return data[i]; }

        __host__ __device__ CU_Vector& operator+=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] += b[i];
            }
            return *this;
        }

        template <class S>
        __host__ __device__ CU_Vector& operator*=(const S& scalar) {
            for(int i=0; i<N; i++) {
                data[i] *= scalar;
            }
            return *this;
        }

        __host__ __device__ float norm() {
            float _norm = 0.f;
            for(int i=0; i<N; i++) {
                float tmp = data[i];
                _norm += tmp*tmp;
            }
            return sqrtf(_norm);
        }

    protected:
        int N;
        float data[VecSize];
};

class CU_Vector3f : public CU_Vector<3> 
{
    using CU_Vector::CU_Vector;

    public:
        __host__ __device__ CU_Vector3f(float x, float y, float z) {
            data[0] = x;
            data[1] = y;
            data[2] = z;
        }
};

template <class T>
__host__ __device__ T operator+(T &a, T &b) {
    T result;
    for(int i = 0; i < a.get_size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}