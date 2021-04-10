#include "cuda/scene.cuh"

#define PI 3.14159265358979

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

void raytrace_spheres(Sphere* spheres, size_t sphere_count, int* visible, CU_Vector3f* vertices, CU_Vector3f* normals, CU_Vector3f* image, Camera* camera);

Scene::Scene() {
    camera = (Camera*)malloc(sizeof(struct Camera));
    (camera->E).set_identity();

    // this->camera->E = E;
    set_camera_intrinsics(60.f, 512, 512);

    size_t vertex_count = this->camera->width * this->camera->height;
    visible = (int*)malloc(vertex_count*sizeof(int));
    vertices = (CU_Vector3f*)malloc(vertex_count*sizeof(CU_Vector3f));
    normals = (CU_Vector3f*)malloc(vertex_count*sizeof(CU_Vector3f));
    image = (CU_Vector3f*)malloc(vertex_count*sizeof(CU_Vector3f));
}

Scene::~Scene() {
    free(camera);
    free(visible);
    free(vertices);
    free(normals);
    free(image);
}

void Scene::create_default_scene() {

    // Initialize camera position
    rotate_camera(0.f, 0.f, 0.f);
    transform_camera(0.f, 0.f, 55.f);

    printf("Camera extrinsics:\n");
    printf("[%.3f, %.3f, %.3f, %.3f]\n", camera->E(0,0), camera->E(0,1), camera->E(0,2), camera->E(0,3));
    printf("[%.3f, %.3f, %.3f, %.3f]\n", camera->E(1,0), camera->E(1,1), camera->E(1,2), camera->E(1,3));
    printf("[%.3f, %.3f, %.3f, %.3f]\n", camera->E(2,0), camera->E(2,1), camera->E(2,2), camera->E(2,3));
    printf("[%.3f, %.3f, %.3f, %.3f]\n", camera->E(3,0), camera->E(3,1), camera->E(3,2), camera->E(3,3));


    spheres.clear();

    add_sphere(CU_Vector3f(0.f, 0.f, 0.f), 10.f, CU_Vector3f(0.2, 0.2, 0.2));         // center sphere
    add_sphere(CU_Vector3f(0.f, 0.f, -1000.f), 940.f, CU_Vector3f(0.f, 1.f, 0.f));    // green sphere
    add_sphere(CU_Vector3f(0.f, -1000.f, 0.f), 990.f, CU_Vector3f(0.f, 0.f, 1.f));    // blue sphere
    add_sphere(CU_Vector3f(0.f, 0.f, 1000.f), 940.f, CU_Vector3f(0.f, 0.f, 0.f));       
    add_sphere(CU_Vector3f(0.f, 1000.f, 0.f), 940.f, CU_Vector3f(1.f, 0.f, 0.f));
}

void Scene::render() {
    raytrace_spheres(&spheres[0], spheres.size(), visible, vertices, normals, image, camera);

    size_t num_pixels = camera->width*camera->height;

    std::vector<unsigned char> image_rgb(3*num_pixels, 0);

    for(size_t i = 0; i < num_pixels; i++) {
        image_rgb[3*i + 0] = (unsigned char)(255*image[i][0]);
        image_rgb[3*i + 1] = (unsigned char)(255*image[i][1]);
        image_rgb[3*i + 2] = (unsigned char)(255*image[i][2]);
    }
    stbi_write_png("image.png", camera->width, camera->height, 3, &image_rgb[0], 0);

    for(int i = 0; i < 10; i++) {
        printf("Vertex on sphere %d: (%.3f, %.3f, %.3f)\n", visible[i], vertices[i][0], vertices[i][1], vertices[i][2]);
    }
}

void Scene::set_camera_intrinsics(float fov, size_t width, size_t height) {
    CU_Matrix<3> K;
    camera->K[0*3 + 0] = width / (2*tanf(PI * fov / 360.f));    // f_x
    camera->K[1*3 + 1] = width / (2*tanf(PI * fov / 360.f));    // f_y
    camera->K[2*3 + 2] = 1.f;                                   // 1

    camera->K[0*3 + 2] = width / 2;     // c_x
    camera->K[1*3 + 2] = height / 2;    // c_y

    camera->width = width;
    camera->height = height;
}

void Scene::rotate_camera(float alpha, float beta, float gamma) {
    CU_Matrix<4> Rx;
    CU_Matrix<4> Ry;
    CU_Matrix<4> Rz;

    Rx.set_identity();
    Ry.set_identity();
    Rz.set_identity();

    Rx(1, 1) = cosf(alpha*PI/180);
    Rx(1, 2) = -sinf(alpha*PI/180);
    Rx(2, 1) = -Rx(1, 2);
    Rx(2, 2) = Rx(1, 1);

    Ry(0, 0) = cosf(beta*PI/180);
    Ry(0, 2) = sinf(beta*PI/180);
    Ry(2, 0) = -Ry(0, 2);
    Ry(2, 2) = Ry(0, 0);

    Rz(0, 0) = cosf(gamma*PI/180);
    Rz(0, 1) = -sinf(gamma*PI/180);
    Rz(1, 0) = -Rz(0, 1);
    Rz(1, 1) = Rz(0, 0);

    CU_Matrix<4> R = Rz * Ry * Rx;
    camera->E = R*camera->E;
}

void Scene::transform_camera(float x, float y, float z) {
    camera->E[0*4 + 3] += x;
    camera->E[1*4 + 3] += y;
    camera->E[2*4 + 3] += z;
}

void Scene::add_sphere(CU_Vector3f pos, float radius, CU_Vector3f color) {
    Sphere s;
    s.pos = pos;
    s.radius = radius;
    s.color = color;
    // memcpy(s.color, color, sizeof(CU_Vector3f));
    spheres.push_back(s);
}

std::vector<Sphere> Scene::get_spheres() { return spheres; }

size_t Scene::get_sphere_count() { return spheres.size(); }
Camera* Scene::get_camera() { return camera; }