#include "cuda/vector.cuh"
#include "cuda/matrix.cuh"

struct Sphere {
    CU_Vector3f pos;
    float radius;
    CU_Vector3f color;
};

struct Camera {
    CU_Matrix<4> E;         // camera extrinsics
    CU_Matrix<4> E_inv;     // inv camera extrinsics
    CU_Matrix<3> K;         // camera intrinsics
    size_t width = 512;
    size_t height = 512;
};