#include <stdio.h>
#include <vector>
#include <math.h>

class Vector4f;

class Vector {
    public:
        explicit Vector(double x = 0.f, double y = 0.f, double z = 0.f) {
            coords[0] = x;
            coords[1] = y;
            coords[2] = z;
        };

        Vector& operator+=(const Vector& b) {
            coords[0] += b[0];
            coords[1] += b[1];
            coords[2] += b[2];
            return *this;
        }

        template <typename T>
        Vector& operator*=(const T& scalar) {
            coords[0] *= scalar;
            coords[1] *= scalar;
            coords[2] *= scalar;
            return *this;
        }

        template <typename T>
        Vector& operator/=(const T& scalar) {
            this *= 1.f/scalar;
            return *this;
        }

        Vector4f get4dim() {
            return Vector4f(coords[0], coords[1], coords[2], 1.f);
        }

        const double& operator[](int i) const { return coords[i]; }
        double& operator[](int i) { return coords[i]; }

        const double norm() {
            return sqrt(pow(coords[0], 2)+pow(coords[1], 2)+pow(coords[2], 2));
        }

        void normalize() {
            const double _norm = this->norm();

            coords[0] /= _norm;
            coords[1] /= _norm;
            coords[2] /= _norm;
        }

    private:
        double coords[3];
};

class Vector4f {
    public:
        explicit Vector4f(double x, double y, double z, double w) {
            coords[0] = x;
            coords[1] = y;
            coords[2] = z;
            coords[3] = w;
        }

        Vector get3dim() {
            return Vector(coords[0], coords[1], coords[2]);
        }

    private:
        double coords[4];
};

// Overload vector operations
Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
};

template <typename T>
Vector operator*(const T& scalar, const Vector& a) {
    return Vector(scalar*a[0], scalar*a[1], scalar*a[2]);
};

template <typename T>
Vector operator*(const Vector& a, const T& scalar) {
    return scalar*a;
};

// Define dot and cross products
double dot(const Vector& a, const Vector& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
};

Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]);
}