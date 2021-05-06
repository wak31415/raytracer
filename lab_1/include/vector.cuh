#ifndef CU_VECTOR_CUH
#define CU_VECTOR_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

template <size_t VecSize, typename T>
class CU_Vector {
    public:
        CUDA_CALLABLE_MEMBER CU_Vector() : N(VecSize) {
            memset(data, 0, sizeof(T)*N);
        }

        CUDA_CALLABLE_MEMBER CU_Vector(T* data) : N(VecSize) {
            memcpy(this->data, data, VecSize*sizeof(T));
        }

        CUDA_CALLABLE_MEMBER CU_Vector(T x, T y, T z) : N(3) {
            data[0] = x;
            data[1] = y;
            data[2] = z;
        }

        CUDA_CALLABLE_MEMBER size_t get_size() { return N; }

        CUDA_CALLABLE_MEMBER const T& operator[](int i) const { return data[i]; }
        CUDA_CALLABLE_MEMBER T& operator[](int i) { return data[i]; }


        // Defining Math Operators
        CUDA_CALLABLE_MEMBER CU_Vector& operator+=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] += b[i];
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER CU_Vector& operator-=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] -= b[i];
            }
            return *this;
        }

        template <class S>
        CUDA_CALLABLE_MEMBER CU_Vector& operator*=(const S& scalar) {
            for(int i=0; i<N; i++) {
                data[i] *= scalar;
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER CU_Vector& operator*=(const CU_Vector& b) {
            for(int i=0; i<N; i++) {
                data[i] *= b[i];
            }
            return *this;
        }

        template <class S>
        CUDA_CALLABLE_MEMBER CU_Vector& operator/=(const S& scalar) {
            this *= 1.f/scalar;
            return *this;
        }

        CUDA_CALLABLE_MEMBER T min() {
            T _min = data[0];
            for (size_t i = 0; i < N; i++)
            {
                if(data[i] < _min) _min = data[i];
            }
            return _min; 
        }

        CUDA_CALLABLE_MEMBER unsigned int argmin_abs() {
            T _min = data[0];
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

        CUDA_CALLABLE_MEMBER T norm() {
            T _norm = 0.f;
            for(int i=0; i<N; i++) {
                T tmp = data[i];
                _norm += tmp*tmp;
            }
            return sqrtf(_norm);
        }

        CUDA_CALLABLE_MEMBER void normalize() {
            const T _norm = this->norm();
            for (int i = 0; i < N; i++)
                data[i] /= _norm;
        }

        CUDA_CALLABLE_MEMBER CU_Vector cross(const CU_Vector<3, float> &other) {
            // if (N!=3) { return CU_Vector(); }
            CU_Vector<3, float> cross_prod;
            cross_prod[0] = data[1]*other[2] - data[2]*other[1];
            cross_prod[1] = data[2]*other[0] - data[0]*other[2];
            cross_prod[2] = data[0]*other[1] - data[1]*other[0];
            return cross_prod;
        };

    protected:
        int N;
        T data[VecSize];
};

typedef CU_Vector<3, float> CU_Vector3f;
typedef CU_Vector<3, int> CU_Vector3i;
typedef CU_Vector<2, int> CU_Vector2i;


// Overload vector operations
template <size_t VecSize, typename T>
CUDA_CALLABLE_MEMBER CU_Vector<VecSize, T> operator+(CU_Vector<VecSize, T> a, CU_Vector<VecSize, T> b) {
    CU_Vector<VecSize, T> result;
    for(int i = 0; i < VecSize; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
};

template <size_t VecSize, typename T>
CUDA_CALLABLE_MEMBER CU_Vector<VecSize, T> operator-(CU_Vector<VecSize, T> a, CU_Vector<VecSize, T> b) {
    CU_Vector<VecSize, T> result;
    for(int i = 0; i < VecSize; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
};

template <size_t VecSize, typename T, typename S>
CUDA_CALLABLE_MEMBER CU_Vector<VecSize, T> operator*(const S &scalar, const CU_Vector<VecSize, T> &v) {
    CU_Vector<VecSize, T> result;
    for (int i = 0; i < VecSize; i++)
        result[i] = scalar * v[i];
    
    return result;
};

template <size_t VecSize, typename T, typename S>
CUDA_CALLABLE_MEMBER CU_Vector<VecSize, T> operator*(const CU_Vector<VecSize, T> &v, const S &scalar) {
    return scalar * v;
};

// element-wise multiplication
template <size_t VecSize, typename T>
CUDA_CALLABLE_MEMBER CU_Vector<VecSize, T> operator*(const CU_Vector<VecSize, T> &a, const CU_Vector<VecSize, T> &b) {
    T data[VecSize];
    for (int i = 0; i < VecSize; i++)
        data[i] = a[i] * b[i];
    
    return CU_Vector<VecSize, T>(data);
};

// Define dot and cross products
template <size_t VecSize, typename T>
CUDA_CALLABLE_MEMBER float dot(const CU_Vector<VecSize, T>& a, const CU_Vector<VecSize, T>& b) {
    float dot_prod = 0.f;
    for(int i = 0; i < VecSize; i++) dot_prod += a[i]*b[i];
    return dot_prod;
};

// CUDA_CALLABLE_MEMBER CU_Vector<3, float> cross(const CU_Vector<3, float> &a, const CU_Vector<3, float> &b) {
//     return CU_Vector<3, float>(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]);
// };

#endif