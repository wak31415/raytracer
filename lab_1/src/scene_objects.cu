#include <iostream>
#include "scene_objects.cuh"

Object::Object(Material material) {
    this->material = material;
}

Sphere::Sphere() {};
Sphere::Sphere(CU_Vector3f pos, float radius, Material material) : material(material), pos(pos), radius(radius) {}
// Sphere::~Sphere() {};

// __host__ __device__ CU_Vector3f Sphere::get_normal(CU_Vector3f P) {
//     CU_Vector3f tmp = P - pos;
//     tmp.normalize();
//     return tmp;
// }

// __host__ __device__ CU_Vector3f Sphere::get_position() {
//     return pos;
// }

// __host__ __device__ float Sphere::get_radius() {
//     return radius;
// }

// __host__ __device__ float Sphere::get_intersection(CU_Vector3f start, CU_Vector3f ray) {
//     CU_Vector3f O_C = start - pos;

//     float ray_dot_O_C = dot(ray, O_C);
//     float O_C_norm = O_C.norm();
//     float delta = ray_dot_O_C*ray_dot_O_C - O_C_norm*O_C_norm + radius*radius;

//     if (delta >= 0) {
//         float t;
//         float t1 = - ray_dot_O_C - sqrtf(delta);
//         float t2 = - ray_dot_O_C + sqrtf(delta);

//         if (t2 >= 0) {
//             t = t1 >= 0 ? t1 : t2;
//             return t;
//         }
//     }
//     return -1.f;
// };