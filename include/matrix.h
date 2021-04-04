#include <stdio.h>
#include <math.h>
#include "vector.h"

class Matrix {
    public:
        explicit Matrix(double* data, int N) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                    coords[i*N + j] = data[i*N + j];
                }
            }
        };

        Matrix& operator+=(const Matrix& B) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                    coords[i*N + j] += B[i*N + j];
                }
            }
            return *this;
        }

        template <typename T>
        Matrix& operator*=(const T& scalar) {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                    coords[i*N + j] *= scalar;
                }
            }
            return *this;
        }

        template <typename T>
        Matrix& operator/=(const T& scalar) {
            this *= 1.f/scalar;
            return *this;
        }

        const double& operator[](int i) const { return coords[i]; }
        double& operator[](int i) { return coords[i]; }

    private:
        double* coords;
};

Vector4f operator*(const Matrix M, const Vector4f x) {
    Vector4f result(0.f, 0.f, 0.f, 0.f);
    int N = 4;

    for(int i = 0; i < N; i++) {
        double tmp = 0.f;
        for(int j = 0; j < N; j++) {
            tmp += M[i*N + j];
        }
        result[i] = tmp;
    }

    return result;
}