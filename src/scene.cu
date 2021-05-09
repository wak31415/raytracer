#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cmath>

#include "scene.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
                      BoundingBox* bounding_boxes,
                      size_t vertex_count,
                      size_t obj_count,
                      Light* lights, 
                      size_t light_count,
                      CU_Vector3f* image, 
                      Camera* camera, 
                      size_t seed
);

CU_Matrix<4> rotation_matrix(float alpha, float beta, float gamma) {
    CU_Matrix<4> Rx;
    CU_Matrix<4> Ry;
    CU_Matrix<4> Rz;

    Rx.set_identity();
    Ry.set_identity();
    Rz.set_identity();

    Rx(1, 1) = cosf(alpha*M_PI/180);
    Rx(1, 2) = -sinf(alpha*M_PI/180);
    Rx(2, 1) = -Rx(1, 2);
    Rx(2, 2) = Rx(1, 1);

    Ry(0, 0) = cosf(beta*M_PI/180);
    Ry(0, 2) = sinf(beta*M_PI/180);
    Ry(2, 0) = -Ry(0, 2);
    Ry(2, 2) = Ry(0, 0);

    Rz(0, 0) = cosf(gamma*M_PI/180);
    Rz(0, 1) = -sinf(gamma*M_PI/180);
    Rz(1, 0) = -Rz(0, 1);
    Rz(1, 1) = Rz(0, 0);

    CU_Matrix<4> R = Rz * Ry * Rx;
    return R;
}

CU_Matrix<4> rotation_matrix(CU_Vector3f rotation) {
    return rotation_matrix(rotation[0], rotation[1], rotation[2]);
}

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

void Scene::render(std::string imagepath, size_t seed) {
    raytrace_spheres(&spheres[0], spheres.size(), &triangles[0], triangles.size(), &vertices[0], &normals[0], &bounding_boxes[0], vertices.size(), bounding_boxes.size(), &lights[0], lights.size(), image, camera, seed);

    size_t num_pixels = camera->width*camera->height;

    std::vector<unsigned char> image_rgb(3*num_pixels, 0);

    for(size_t i = 0; i < num_pixels; i++) {
        image_rgb[3*i + 0] = (unsigned char)(fminf(image[i][0], 255));
        image_rgb[3*i + 1] = (unsigned char)(fminf(image[i][1], 255));
        image_rgb[3*i + 2] = (unsigned char)(fminf(image[i][2], 255));
    }
    stbi_write_png(imagepath.c_str(), camera->width, camera->height, 3, &image_rgb[0], 0);
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
    camera->K[0*3 + 0] = width / (2*tanf(M_PI * fov / 360.f));    // f_x
    camera->K[1*3 + 1] = width / (2*tanf(M_PI * fov / 360.f));    // f_y
    camera->K[2*3 + 2] = 1.f;                                   // 1

    camera->K[0*3 + 2] = width / 2;     // c_x
    camera->K[1*3 + 2] = height / 2;    // c_y

    camera->width = width;
    camera->height = height;
}

void Scene::rotate_camera(float alpha, float beta, float gamma) {
    camera->E = rotation_matrix(alpha, beta, gamma)*camera->E;
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

void Scene::add_object(std::string inputfile, CU_Vector3f translation, CU_Vector3f scale, CU_Vector3f rotation, Material material) {

    CU_Matrix<4> transform;
    transform.set_identity();
    transform.scale(scale);
    transform = rotation_matrix(rotation) * transform;
    transform.translate(translation);


    // Largely copied from reference usage: https://github.com/tinyobjloader/tinyobjloader

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(inputfile)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        BoundingBox bounding_box;
        bounding_box._max = CU_Vector3f(-INFINITY, -INFINITY, -INFINITY);
        bounding_box._min = CU_Vector3f(INFINITY, INFINITY, INFINITY);

        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            bool has_normal = true;
            CU_Vector3f face_normals[fv];

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                CU_Vector3f vtx = CU_Vector3f((float)vx, (float)vy, (float)vz);
                vtx = transform * vtx;

                for (size_t i = 0; i < 3; i++)
                {
                    if(vtx[i] < bounding_box._min[i]) { bounding_box._min[i] = vtx[i]; }
                    else if(vtx[i] > bounding_box._max[i]) { bounding_box._max[i] = vtx[i]; }
                }
                
            
                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];

                    face_normals[v][0] = (float)nx;
                    face_normals[v][1] = (float)ny;
                    face_normals[v][2] = (float)nz;
                    
                } else {
                    has_normal = false;
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                    tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            
                vertices.push_back(vtx);
            }
            for (size_t v = 0; v < fv; v++)
            {
                if(has_normal) {
                    CU_Vector3f N = face_normals[v];
                    N = transform.get_rotation() * N;
                    N.normalize();
                    normals.push_back(N);
                } else {
                    CU_Vector3f N = (vertices[index_offset+1]-vertices[index_offset]).cross(vertices[index_offset+2] - vertices[index_offset]);
                    N = transform.get_rotation() * N;
                    N.normalize();
                    normals.push_back(N);
                }
            }

            Triangle t;
            t.v = CU_Vector3i(index_offset, index_offset+1, index_offset+2);
            t.n = CU_Vector3i(index_offset, index_offset+1, index_offset+2);
            t.group = s;

            t.material = material;
            triangles.push_back(t);

            index_offset += fv;
            // per-face material
            // shapes[s].mesh.material_ids[f];
        }
        bounding_boxes.push_back(bounding_box);
    }

    std::cout << "Minimum bound: ";
    for (size_t i = 0; i < 3; i++)
    {
        std::cout << bounding_boxes[0]._min[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "Maximum bound: ";
    for (size_t i = 0; i < 3; i++)
    {
        std::cout << bounding_boxes[0]._max[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Loaded " << bounding_boxes.size() << " objects." << std::endl;

    std::cout << "Loaded " << vertices.size() << " vertices." << std::endl;
    std::cout << "Loaded " << normals.size() << " normals." << std::endl;
    std::cout << "Loaded " << triangles.size() << " triangles." << std::endl;
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