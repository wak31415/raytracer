#include "cuda/scene_objects.cuh"
#include "cuda/projection_helpers.cuh"
#include <stdio.h>

__global__ void raytrace_spheres_kernel(Sphere* spheres, 
                                        size_t sphere_count, 
                                        int* visible, 
                                        CU_Vector3f* vertices, 
                                        CU_Vector3f* normals,
                                        CU_Vector3f* image, 
                                        CU_Matrix<3> cam_rot,
                                        CU_Vector3f camera_pos,
                                        CU_Matrix<3> K,
                                        uint width,
                                        uint height) 
{
    uint u_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint u_y = blockDim.y * blockIdx.y + threadIdx.y;

    uint idx = u_y * width + u_x;

    if (u_x >= width || u_y >= height) 
        return;
    
    if (idx == 0) {
        printf("fx, fy = (%.3f, %.3f)\n", K[0*3+0], K[1*3+1]);
        printf("Color sphere 1: (%.f, %.f, %.f)\n", spheres[1].color[0], spheres[1].color[1], spheres[1].color[2]);
        printf("Camera position: (%.3f, %.3f, %.3f)\n", camera_pos[0], camera_pos[1], camera_pos[2]);
        // printf("Camera extrinsics:\n");
        // printf("fx, fy = (%.3f, %.3f)\n", camera->K[0*3+0], camera->K[1*3+1]);
    }



    // obtain ray direction
    CU_Vector3f ray_dir = pixel_to_camera(u_x, u_y, 1.f, K);
    ray_dir.normalize();
    ray_dir = cam_rot*ray_dir;

    if(idx < 10) {
        printf("Ray direction: (%.3f, %.3f, %.3f)\n", ray_dir[0], ray_dir[1], ray_dir[2]);
    }

    float min_dist = 0.f;
    int min_id = -1;

    for(size_t i = 0; i < sphere_count; i++) {
        CU_Vector3f sphere_to_cam = camera_pos - spheres[i].pos;
        float dot_prod = dot(ray_dir, sphere_to_cam);
        
        float delta = powf(dot_prod, 2.f) - (powf(sphere_to_cam.norm(), 2.f) - powf(spheres[i].radius, 2.f));

        if (delta >= 0) {
            float t;
            float t1 = - dot_prod - sqrtf(delta);
            float t2 = - dot_prod + sqrtf(delta);
            
            t = t1 >= 0 ? t1 : t2;

            if (min_id == -1 || t < min_dist) {
                min_dist = t;
                min_id = i;
            }
        }
    }

    // if(idx < 10) {
    //     printf("Intersecting with sphere %d\n", min_id);
    // }

    visible[idx] = min_id;

    if(min_id >= 0) {
        CU_Vector3f P = camera_pos + min_dist*ray_dir;
        vertices[idx] = P;
        CU_Vector3f tmp = P - spheres[min_id].pos;
        normals[idx] = (1/tmp.norm()) * tmp;
        image[idx] = spheres[min_id].color;
    }
}

void raytrace_spheres(Sphere* spheres, size_t sphere_count, int* visible, CU_Vector3f* vertices, CU_Vector3f* normals, CU_Vector3f* image, Camera* camera) {
    size_t vertex_count = camera->width * camera->height;

    Sphere* d_spheres;
    CU_Matrix<4> d_cam_rot;
    CU_Vector3f d_cam_trans;
    CU_Matrix<3> d_K;
    int* d_visible;
    CU_Vector3f* d_image;
    CU_Vector3f* d_vertices;
    CU_Vector3f* d_normals;

    CU_Matrix<3> cam_rot = camera->E.get_rotation();
    CU_Vector3f cam_trans = camera->E.get_translation();

    cudaMalloc((void**)&d_spheres, sphere_count*sizeof(struct Sphere));
    // cudaMalloc((void**)&d_cam_rot, sizeof(CU_Matrix<3>));
    // cudaMalloc((void**)&d_cam_trans, sizeof(CU_Vector3f));
    // cudaMalloc((void**)&d_K, sizeof(CU_Matrix<3>));
    cudaMalloc((void**)&d_visible, vertex_count*sizeof(int));
    cudaMalloc((void**)&d_image, vertex_count*sizeof(CU_Vector3f));
    cudaMalloc((void**)&d_vertices, vertex_count*sizeof(CU_Vector3f));
    cudaMalloc((void**)&d_normals, vertex_count*sizeof(CU_Vector3f));

    cudaMemcpy(d_spheres, spheres, sphere_count*sizeof(struct Sphere), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cam_rot, cam_rot, sizeof(CU_Matrix<3>), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cam_trans, cam_trans, sizeof(CU_Vector3f), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_K, camera-K, sizeof(CU_Matrix<3>), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid((camera->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (camera->height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    raytrace_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_spheres,
        sphere_count,
        d_visible,
        d_vertices,
        d_normals,
        d_image,
        cam_rot,
        cam_trans,
        camera->K,
        512,
        512
    );
    cudaDeviceSynchronize();

    cudaMemcpy(visible, d_visible, vertex_count*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(image, d_image, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(vertices, d_vertices, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(normals, d_normals, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);

    cudaFree(d_spheres);
    // cudaFree(d_cam_rot);
    // cudaFree(d_cam_trans);
    // cudaFree(d_K);
    cudaFree(d_visible);
    cudaFree(d_image);
    cudaFree(d_vertices);
    cudaFree(d_normals);
}