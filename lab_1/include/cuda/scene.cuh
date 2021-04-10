#ifndef SCENE_CUH
#define SCENE_CUH

#include <vector>
#include "cuda/scene_objects.cuh"

class Scene {
    public:
        Scene();
        ~Scene();

        void create_default_scene();

        void render();

        void set_camera_intrinsics(float fov, size_t width, size_t height);

        void rotate_camera(float alpha, float beta, float gamma);
        void transform_camera(float x, float y, float z);

        void add_sphere(CU_Vector3f pos, float radius, CU_Vector3f color);
        std::vector<Sphere> get_spheres();
        size_t get_sphere_count();

        Camera* get_camera();

    private:
        std::vector<Sphere> spheres;
        Camera* camera;
        int* visible;
        CU_Vector3f* vertices;
        CU_Vector3f* normals;
        CU_Vector3f* image;
};

#endif