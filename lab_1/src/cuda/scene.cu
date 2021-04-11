#include <nlohmann/json.hpp>
#include <fstream>
#include <stdio.h>

#include "cuda/scene.cuh"

#define PI 3.14159265358979

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

using json = nlohmann::json;

void raytrace_spheres(Sphere* spheres, 
                      size_t sphere_count, 
                      Light* lights, 
                      size_t light_count, 
                      int* visible, 
                      CU_Vector3f* vertices, 
                      CU_Vector3f* normals, 
                      CU_Vector3f* image, 
                      Camera* camera
);

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

void Scene::load_scene(std::string filepath) {
    camera->E[1*4+1] = -1.f;
    camera->E[2*4+2] = -1.f;

    std::ifstream ifs("../scene.json");
    json jf = json::parse(ifs);

    spheres.clear();

    CU_Vector3f camera_pos(jf["camera"]["pos"][0],
                           jf["camera"]["pos"][1],
                           jf["camera"]["pos"][2]);

    CU_Vector3f camera_rot(jf["camera"]["rotation"][0],
                           jf["camera"]["rotation"][1],
                           jf["camera"]["rotation"][2]);

    rotate_camera(camera_rot);
    transform_camera(camera_pos);

    // Add spheres
    for(int i = 0; i < jf["spheres"].size(); i ++) {
        CU_Vector3f pos(jf["spheres"][i]["pos"][0],
                        jf["spheres"][i]["pos"][1],
                        jf["spheres"][i]["pos"][2]);

        CU_Vector3f color(jf["spheres"][i]["color"][0],
                          jf["spheres"][i]["color"][1],
                          jf["spheres"][i]["color"][2]);

        add_sphere(pos, jf["spheres"][i]["radius"], color);
    }

    // Add lights
    for(int i = 0; i < jf["lights"].size(); i ++) {
        CU_Vector3f pos(jf["lights"][i]["pos"][0],
                        jf["lights"][i]["pos"][1],
                        jf["lights"][i]["pos"][2]);

        add_light(pos, jf["lights"][i]["intensity"]);
    }
}

void Scene::create_default_scene() {

    // Initialize camera position
    camera->E[1*4+1] = -1.f;
    camera->E[2*4+2] = -1.f;
    rotate_camera(0.f, 0.f, 0.f);
    transform_camera(0.f, 0.f, 55.f);

    spheres.clear();

    add_sphere(CU_Vector3f(0.f, 0.f, 0.f), 10.f, CU_Vector3f(0.5, 0.5, 0.5));         // center sphere
    add_sphere(CU_Vector3f(0.f, 0.f, -1000.f), 940.f, CU_Vector3f(0.f, 1.f, 0.f));    // green sphere
    add_sphere(CU_Vector3f(0.f, -1000.f, 0.f), 990.f, CU_Vector3f(0.f, 0.f, 1.f));    // blue sphere
    add_sphere(CU_Vector3f(0.f, 0.f, 1000.f), 940.f, CU_Vector3f(1.f, 0.f, 1.f));       // magenta sphere
    add_sphere(CU_Vector3f(0.f, 1000.f, 0.f), 940.f, CU_Vector3f(1.f, 0.f, 0.f));

    add_light(CU_Vector3f(-20.f, 20.f, 30.f), 100000.f);
}

void Scene::render() {
    raytrace_spheres(&spheres[0], spheres.size(), &lights[0], lights.size(), visible, vertices, normals, image, camera);

    size_t num_pixels = camera->width*camera->height;

    std::vector<unsigned char> image_rgb(3*num_pixels, 0);

    for(size_t i = 0; i < num_pixels; i++) {
        image_rgb[3*i + 0] = (unsigned char)(255*image[i][0]);
        image_rgb[3*i + 1] = (unsigned char)(255*image[i][1]);
        image_rgb[3*i + 2] = (unsigned char)(255*image[i][2]);
    }
    stbi_write_png("image.png", camera->width, camera->height, 3, &image_rgb[0], 0);
}

void Scene::set_camera_intrinsics(float fov, size_t width, size_t height) {
    CU_Matrix<3> K;
    // f: focal length
    // W: sensor width
    // H: sensor height
    // w: image width (pixels)
    // h: image height (pixels)
    // f_x = f*w/W
    // f_y = f*h/H
    camera->K[0*3 + 0] = width / (2*tanf(PI * fov / 360.f));    // f_x
    camera->K[1*3 + 1] = height / (2*tanf(PI * fov / 360.f));    // f_y
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

void Scene::rotate_camera(CU_Vector3f rotation) {
    rotate_camera(rotation[0], rotation[1], rotation[2]);
}

void Scene::transform_camera(float x, float y, float z) {
    camera->E[0*4 + 3] += x;
    camera->E[1*4 + 3] += y;
    camera->E[2*4 + 3] += z;
}

void Scene::transform_camera(CU_Vector3f direction) {
    camera->E[0*4 + 3] += direction[0];
    camera->E[1*4 + 3] += direction[1];
    camera->E[2*4 + 3] += direction[2];
}

void Scene::add_sphere(CU_Vector3f pos, float radius, CU_Vector3f color) {
    Sphere s;
    s.pos = pos;
    s.radius = radius;
    s.color = color;
    spheres.push_back(s);
}

void Scene::add_light(CU_Vector3f pos, float intensity) {
    Light l;
    l.pos = pos;
    l.I = intensity;
    lights.push_back(l);
}

std::vector<Sphere> Scene::get_spheres() { return spheres; }

size_t Scene::get_sphere_count() { return spheres.size(); }
Camera* Scene::get_camera() { return camera; }