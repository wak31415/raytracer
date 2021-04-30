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

class Object {
    public:
        Object(Material material);

        __host__ __device__ Material get_material() { return material; };
        virtual __host__ __device__ CU_Vector3f get_normal(CU_Vector3f P) { return CU_Vector3f(); };
        virtual __host__ __device__ float get_intersection(CU_Vector3f start, CU_Vector3f ray) { return 0.f; };

    protected:
        Material material;
};

class Sphere {
    public:
        Sphere();
        Sphere(CU_Vector3f pos, float radius, Material type);

        // __host__ __device__ CU_Vector3f get_normal(CU_Vector3f P);
        // __host__ __device__ CU_Vector3f get_position();
        // __host__ __device__ float get_radius();

        // __host__ __device__ float get_intersection(CU_Vector3f start, CU_Vector3f ray);
    
        CU_Vector3f pos;
        float radius;
        Material material;
};

// class Triangle : Object {
//     public:
//         Triangle() {};
//         virtual ~Triangle() {};
// };