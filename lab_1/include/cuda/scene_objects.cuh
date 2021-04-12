#include "cuda/vector.cuh"
#include "cuda/matrix.cuh"

struct Sphere {
    CU_Vector3f pos;
    float radius;
    CU_Vector3f color;
    float specularity;
};

struct Camera {
    CU_Matrix<4> E;         // camera extrinsics
    CU_Matrix<4> E_inv;     // inv camera extrinsics
    CU_Matrix<3> K;         // camera intrinsics
    uint width = 1024;
    uint height = 1024;
};

struct Light {
    CU_Vector3f pos;
    float I;
};