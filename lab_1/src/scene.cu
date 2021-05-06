#include <nlohmann/json.hpp>
#include <fstream>
#include <stdio.h>
#include <string>

#include "scene.cuh"

#define PI 3.14159265358979

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

using json = nlohmann::json;

void raytrace_spheres(Sphere* spheres, 
                      size_t sphere_count, 
                      Triangle* triangles,
                      size_t triangle_count,
                      CU_Vector3f* vertices,
                      CU_Vector3f* normals,
                      size_t vertex_count,
                      Light* lights, 
                      size_t light_count,
                      CU_Vector3f* image, 
                      Camera* camera
);

Scene::Scene() {
    camera = (Camera*)malloc(sizeof(struct Camera));
    (camera->E).set_identity();

    // this->camera->E = E;
    set_camera_intrinsics(60.f, 512, 512);

    size_t vertex_count = this->camera->width * this->camera->height;
    image = (CU_Vector3f*)malloc(vertex_count*sizeof(CU_Vector3f));
}

Scene::~Scene() {
    free(camera);
    free(image);
}

void Scene::load_scene(std::string filepath) {
    camera->E[1*4+1] = -1.f;
    camera->E[2*4+2] = -1.f;

    std::ifstream ifs(filepath);
    if(!ifs.is_open()) {
        throw std::invalid_argument("File not found");
    }
    json jf = json::parse(ifs);

    spheres.clear();

    CU_Vector3f camera_pos(jf["camera"]["pos"][0],
                           jf["camera"]["pos"][1],
                           jf["camera"]["pos"][2]);

    CU_Vector3f camera_rot(jf["camera"]["rotation"][0],
                           jf["camera"]["rotation"][1],
                           jf["camera"]["rotation"][2]);

    uint width = (uint)jf["camera"]["width"];
    uint height = (uint)jf["camera"]["height"];
    float fov = jf["camera"]["fov"];

    set_camera_intrinsics(fov, width, height);

    rotate_camera(camera_rot);
    transform_camera(camera_pos);

    camera->num_rays = jf["camera"]["num_rays"];

    // Add spheres
    for(int i = 0; i < jf["spheres"].size(); i ++) {
        CU_Vector3f pos(jf["spheres"][i]["pos"][0],
                        jf["spheres"][i]["pos"][1],
                        jf["spheres"][i]["pos"][2]);

        std::string mat = jf["spheres"][i]["material"];

        CU_Vector3f color;
        material_type material = DIFFUSE;
        float ri(0.f);
        float ro(0.f);

        if(mat == "diffuse") {
            material = DIFFUSE;
            color[0] = jf["spheres"][i]["color"][0];
            color[1] = jf["spheres"][i]["color"][1];
            color[2] = jf["spheres"][i]["color"][2];
        }
        else if (mat == "mirror") {
            material = MIRROR;
        } 
        else if (mat == "glass") {
            material = GLASS;
            ro = jf["spheres"][i]["refractive_index"][0];
            ri = jf["spheres"][i]["refractive_index"][1];
        }

        float R = jf["spheres"][i]["radius"];

        add_sphere(pos, R, color, material, ro, ri);
    }

    // Add objects
    for(int i = 0; i < jf["objects"].size(); i ++) {
        std::string obj_path = jf["objects"][i]["filepath"];

        CU_Vector3f pos(jf["objects"][i]["pos"][0],
                        jf["objects"][i]["pos"][1],
                        jf["objects"][i]["pos"][2]);

        CU_Vector3f scale(jf["objects"][i]["scale"][0],
                          jf["objects"][i]["scale"][1],
                          jf["objects"][i]["scale"][2]);

        CU_Vector3f rotation(jf["objects"][i]["rotation"][0],
                             jf["objects"][i]["rotation"][1],
                             jf["objects"][i]["rotation"][2]);

        std::string mat = jf["objects"][i]["material"];

        CU_Vector3f color;
        material_type mat_type = DIFFUSE;
        float ri(0.f);
        float ro(0.f);

        if(mat == "diffuse") {
            mat_type = DIFFUSE;
            color[0] = jf["objects"][i]["color"][0];
            color[1] = jf["objects"][i]["color"][1];
            color[2] = jf["objects"][i]["color"][2];
        }
        else if (mat == "mirror") {
            mat_type = MIRROR;
        } 
        else if (mat == "glass") {
            mat_type = GLASS;
            ro = jf["objects"][i]["refractive_index"][0];
            ri = jf["objects"][i]["refractive_index"][1];
        }

        Material material;
        material.type = mat_type;
        material.color = color;
        material.ri = ri;
        material.ro = ro;

        add_object(obj_path, pos, scale, rotation, material);
    }

    // Add lights
    for(int i = 0; i < jf["lights"].size(); i ++) {
        CU_Vector3f pos(jf["lights"][i]["pos"][0],
                        jf["lights"][i]["pos"][1],
                        jf["lights"][i]["pos"][2]);

        add_light(pos, jf["lights"][i]["intensity"]);
    }

    size_t vertex_count = this->camera->width * this->camera->height;
    image = (CU_Vector3f*)realloc(image, vertex_count*sizeof(CU_Vector3f));
}

void Scene::render() {
    raytrace_spheres(&spheres[0], spheres.size(), &triangles[0], triangles.size(), &vertices[0], &normals[0], vertices.size(), &lights[0], lights.size(), image, camera);

    size_t num_pixels = camera->width*camera->height;

    std::vector<unsigned char> image_rgb(3*num_pixels, 0);

    for(size_t i = 0; i < num_pixels; i++) {
        image_rgb[3*i + 0] = (unsigned char)(fminf(image[i][0], 255));
        image_rgb[3*i + 1] = (unsigned char)(fminf(image[i][1], 255));
        image_rgb[3*i + 2] = (unsigned char)(fminf(image[i][2], 255));
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

void Scene::add_sphere(CU_Vector3f pos, float radius, CU_Vector3f color, material_type material, float ro, float ri) {
    Material mat;
    mat.type = material;
    mat.color = color;
    mat.ro = ro;
    mat.ri = ri;
    
    Sphere s;
    s.pos = pos;
    s.radius = radius;
    s.material = mat;
    spheres.push_back(s);
}

void Scene::add_object(std::string obj_path, CU_Vector3f pos, CU_Vector3f scale, CU_Vector3f rotation, Material material) {

    Triangle t;
    CU_Vector3f A(-10, 10.f, 22.f);
    CU_Vector3f B(-20, 0.f, 30.f);
    CU_Vector3f C(0.f, 0.f, 30.f);
    CU_Vector3f N = (B - A).cross(C - A);
    N.normalize();

    vertices.push_back(A);
    vertices.push_back(B);
    vertices.push_back(C);
    normals.push_back(N);
    normals.push_back(N);
    normals.push_back(N);

    t.v = CU_Vector3i(0, 1, 2);
    t.n = CU_Vector3i(0, 1, 2);

    t.material = material;
    triangles.push_back(t);
    return;
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