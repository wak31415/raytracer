#ifndef SCENE_CUH
#define SCENE_CUH

#include <vector>
#include <string>
#include "cuda/scene_objects.cuh"

class Scene {
    public:
        Scene();
        ~Scene();

        void load_scene(std::string filepath);

        void render();

        void set_camera_intrinsics(float fov, size_t width, size_t height);

        void rotate_camera(float alpha, float beta, float gamma);
        void rotate_camera(CU_Vector3f rotation);
        void transform_camera(float x, float y, float z);
        void transform_camera(CU_Vector3f direction);

        void add_sphere(CU_Vector3f pos, float radius, CU_Vector3f color, uint material, float ro, float ri);
        void add_light(CU_Vector3f pos, float intensity);
        
        std::vector<Sphere> get_spheres();
        size_t get_sphere_count();

        Camera* get_camera();

    private:
        std::vector<Sphere> spheres;
        std::vector<Light> lights;
        Camera* camera;
        CU_Vector3f* image;
};

#endif