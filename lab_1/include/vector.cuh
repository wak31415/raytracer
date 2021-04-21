#ifndef CU_VECTOR_CUH
#define CU_VECTOR_CUH

template <size_t VecSize>
class CU_Vector {
    public:
        __host__ __device__ CU_Vector() : N(VecSize) {
            memset(data, 0, sizeof(float)*N);
        }

        __host__ __device__ CU_Vector(float* data) : N(VecSize) {
            memcpy(this->data, data, VecSize*sizeof(float));
        }

        __host__ __device__ CU_Vector(float x, float y, float z) : N(3) {
            data[0] = x;
            data[1] = y;
            data[2] = z;
        }

        __host__ __device__ size_t get_size() { return N; }

        __host__ __device__ const float& operator[](int i) const { return data[i]; }
        __host__ __device__ float& operator[](int i) { return data[i]; }


        // Defining Math Operators
        __host__ __device__ CU_Vector& operator+=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] += b[i];
            }
            return *this;
        }

        __host__ __device__ CU_Vector& operator-=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] -= b[i];
            }
            return *this;
        }

        template <class T>
        __host__ __device__ CU_Vector& operator*=(const T& scalar) {
            for(int i=0; i<N; i++) {
                data[i] *= scalar;
            }
            return *this;
        }

        __host__ __device__ CU_Vector& operator*=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] *= b[i];
            }
            return *this;
        }

        template <class T>
        __host__ __device__ CU_Vector& operator/=(const T& scalar) {
            this *= 1.f/scalar;
            return *this;
        }

        __host__ __device__ float min() {
            float _min = data[0];
            for (size_t i = 0; i < N; i++)
            {
                if(data[i] < _min) _min = data[i];
            }
            return _min; 
        }

        __host__ __device__ unsigned int argmin_abs() {
            float _min = data[0];
            unsigned int _arg = 0;
            for (size_t i = 0; i < N; i++)
            {
                if(abs(data[i]) < _min) {
                    _min = abs(data[i]);
                    _arg = i;
                }
            }
            return _arg;
        }

        __host__ __device__ float norm() {
            float _norm = 0.f;
            for(int i=0; i<N; i++) {
                float tmp = data[i];
                _norm += tmp*tmp;
            }
            return sqrtf(_norm);
        }

        __host__ __device__ void normalize() {
            const float _norm = this->norm();
            for (int i = 0; i < N; i++)
                data[i] /= _norm;
        }

        __host__ __device__ CU_Vector cross(const CU_Vector<3> &other) {
            // if (N!=3) { return CU_Vector(); }
            CU_Vector<3> cross_prod;
            cross_prod[0] = data[1]*other[2] - data[2]*other[1];
            cross_prod[1] = data[2]*other[0] - data[0]*other[2];
            cross_prod[2] = data[0]*other[1] - data[1]*other[0];
            return cross_prod;
        };

    protected:
        int N;
        float data[VecSize];
};

typedef CU_Vector<3> CU_Vector3f;


// Overload vector operations
template <size_t VecSize>
__host__ __device__ CU_Vector<VecSize> operator+(CU_Vector<VecSize> a, CU_Vector<VecSize> b) {
    CU_Vector<VecSize> result;
    for(int i = 0; i < VecSize; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
};

template <size_t VecSize>
__host__ __device__ CU_Vector<VecSize> operator-(CU_Vector<VecSize> a, CU_Vector<VecSize> b) {
    CU_Vector<VecSize> result;
    for(int i = 0; i < VecSize; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
};

template <size_t VecSize, class T>
__host__ __device__ CU_Vector<VecSize> operator*(const T &scalar, const CU_Vector<VecSize> &v) {
    float data[VecSize];
    for (int i = 0; i < VecSize; i++)
        data[i] = scalar * v[i];
    
    return CU_Vector<VecSize>(data);
};

template <size_t VecSize, class T>
__host__ __device__ CU_Vector<VecSize> operator*(const CU_Vector<VecSize> &v, const T &scalar) {
    return scalar * v;
};

// element-wise multiplication
template <size_t VecSize>
__host__ __device__ CU_Vector<VecSize> operator*(const CU_Vector<VecSize> &a, const CU_Vector<VecSize> &b) {
    float data[VecSize];
    for (int i = 0; i < VecSize; i++)
        data[i] = a[i] * b[i];
    
    return CU_Vector<VecSize>(data);
};

// Define dot and cross products
template <size_t VecSize>
__host__ __device__ float dot(const CU_Vector<VecSize>& a, const CU_Vector<VecSize>& b) {
    float dot_prod = 0.f;
    for(int i = 0; i < VecSize; i++) dot_prod += a[i]*b[i];
    return dot_prod;
};

// __host__ __device__ CU_Vector<3> cross(const CU_Vector<3> &a, const CU_Vector<3> &b) {
//     return CU_Vector<3>(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]);
// };

#endif