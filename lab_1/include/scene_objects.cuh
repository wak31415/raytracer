#include "vector.cuh"
#include "matrix.cuh"

enum material_type {
    DIFFUSE = 0,
    MIRROR,
    GLASS,
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

struct Material {
    material_type type;
    CU_Vector3f color;
    float ro; // refractive index outside
    float ri; // refractive index inside
};

struct Sphere {
    CU_Vector3f pos;
    float radius;
    Material material;
};

struct Triangle {
    CU_Vector3i v;
    CU_Vector3i n;
    CU_Vector3i uv;
    int group;
    Material material;
};
