#include "cuda/vector.cuh"
#include "cuda/matrix.cuh"

enum materials {
    DIFFUSE = 0,
    MIRROR,
    GLASS,
};

struct Sphere {
    CU_Vector3f pos;
    float radius;
    CU_Vector3f color;
    uint material;
    float ro; // refractive index outside
    float ri; // refractive index inside
};

struct Camera {
    CU_Matrix<4> E;         // camera extrinsics
    CU_Matrix<4> E_inv;     // inv camera extrinsics
    CU_Matrix<3> K;         // camera intrinsics
    uint width = 1024;
    uint height = 1024;
    uint num_rays = 1;
};

struct Light {
    CU_Vector3f pos;
    float I;
};