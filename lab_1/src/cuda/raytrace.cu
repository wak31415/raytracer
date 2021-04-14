#include "cuda/scene_objects.cuh"
#include "cuda/projection_helpers.cuh"
#include <stdio.h>

#define PI 3.14159265358979
#define GAMMA 2.2
#define MAX_RAY_DEPTH 5

__device__ CU_Vector3f clamp_color(CU_Vector3f color) {
    CU_Vector3f res;
    res[0] = fminf(color[0], 1.f);
    res[1] = fminf(color[1], 1.f);
    res[2] = fminf(color[2], 1.f);
    return res;
}

__device__ CU_Vector3f gamma_correct(CU_Vector3f color) {
    CU_Vector3f res;
    float exponent = 1.f/GAMMA;
    res[0] = powf(color[0], exponent);
    res[1] = powf(color[1], exponent);
    res[2] = powf(color[2], exponent);
    return res;
}

__device__ CU_Vector3f get_intersection(Sphere* spheres, 
                                        size_t sphere_count, 
                                        CU_Vector3f ray, 
                                        CU_Vector3f start, 
                                        int* intersect_id) 
{
    float min_dist = 0.f;

    for(size_t i = 0; i < sphere_count; i++) {
        CU_Vector3f O_C = start - spheres[i].pos;

        float ray_dot_O_C = dot(ray, O_C);
        float delta = powf(ray_dot_O_C, 2.f) - powf(O_C.norm(), 2.f) + powf(spheres[i].radius, 2.f);

        if (delta >= 0) {
            float t;
            float t1 = - ray_dot_O_C - sqrtf(delta);
            float t2 = - ray_dot_O_C + sqrtf(delta);

            if (t2 >= 0) {
                t = t1 >= 0 ? t1 : t2;

                if (*intersect_id < 0 || t < min_dist) {
                    min_dist = t;
                    *intersect_id = i;
                }
            }
        }
    }
    if (min_dist == 0.f) return CU_Vector3f();
    return start + min_dist*ray;
}

__device__ bool is_visible(Sphere* spheres, size_t sphere_count, CU_Vector3f origin, CU_Vector3f target) {
    int intersect_id = -1;
    CU_Vector3f ray = target - origin;
    ray.normalize();
    CU_Vector3f P = get_intersection(spheres, sphere_count, ray, origin, &intersect_id);
    if ((P - origin).norm() < (target - origin).norm() && intersect_id >= 0) return false;
    return true;
}

__device__ CU_Vector3f reflected_direction(CU_Vector3f ray, CU_Vector3f normal) {
    return ray - 2 * dot(ray, normal) * normal;
}

__device__ CU_Vector3f get_color(Sphere* spheres, 
                                 size_t sphere_count, 
                                 Light* lights,
                                 size_t light_count,
                                 CU_Vector3f start, 
                                 CU_Vector3f ray) 
{
    int depth = 0;
    while(depth < MAX_RAY_DEPTH) {
        // if(depth >= MAX_RAY_DEPTH) return CU_Vector3f();
        int intersect_id = -1;
        CU_Vector3f P = get_intersection(spheres, sphere_count, ray, start, &intersect_id);
        
        if(intersect_id >= 0) {
            CU_Vector3f sphere_pos = spheres[intersect_id].pos;
            CU_Vector3f sphere_color = spheres[intersect_id].color;
            int material = spheres[intersect_id].material;

            CU_Vector3f tmp = P - sphere_pos;
            CU_Vector3f N = (1/tmp.norm()) * tmp;

            // Diffuse
            if(material == DIFFUSE) {
                // Normalized vector point --> light
                CU_Vector3f S_P = lights[0].pos - P;
                float d = S_P.norm();
                CU_Vector3f w_i = 1.f/d * S_P;

                float N_wi_dot = max(dot(N, w_i), 0.f);

                // check if the light is visible from P
                bool P_visible = is_visible(spheres, sphere_count, P+0.001*N, lights[0].pos);

                CU_Vector3f L = clamp_color(lights[0].I / (4*powf(PI*d, 2.f)) * sphere_color * P_visible * N_wi_dot);

                return gamma_correct(L);
            }

            // Mirror
            else if(material == MIRROR) {
                CU_Vector3f reflected_ray = reflected_direction(ray, N);
                start = P+0.01*N;
                ray = reflected_ray;
            }

            // Glass 
            else if(material == GLASS) {
                float ro = spheres[intersect_id].ro;
                float ri = spheres[intersect_id].ri;

                CU_Vector3f wt_T;
                CU_Vector3f wt_N;

                float wi_N_dot = dot(ray, N);

                // ray coming from the inside
                if(wi_N_dot > 0) {
                    float tmp1 = ro;
                    ro = ri;
                    ri = tmp1;
                    N = -1.f * N;
                    wi_N_dot = dot(ray, N);
                }

                float tmp = 1.f - (ro/ri)*(ro/ri)*(1.f - wi_N_dot*wi_N_dot);

                if(tmp < 0) {
                    // Total internal reflection
                    CU_Vector3f reflected_ray = reflected_direction(ray, N);
                    
                    start = P + 0.01*N;
                    ray = reflected_ray;
                } else {
                    wt_T = ro / ri * (ray - wi_N_dot*N);
                    wt_N = - sqrtf(tmp)*N;
                    CU_Vector3f wt = wt_T + wt_N;

                    start = P - 0.01*N;
                    ray = wt;
                    // n1 = n2;
                }
            }
            depth = depth + 1;
        }
        else {
            return CU_Vector3f();
        }
    }
    return CU_Vector3f(0.5, 0.5, 0.5);


    
}

// template<>
// __device__ CU_Vector3f get_color<MAX_RAY_DEPTH>(Sphere* spheres, 
//                                                 size_t sphere_count, 
//                                                 Light* lights,
//                                                 size_t light_count,
//                                                 CU_Vector3f start, 
//                                                 CU_Vector3f ray)
// {
//     return CU_Vector3f();
// }

__global__ void raytrace_spheres_kernel(Sphere* spheres, 
                                        size_t sphere_count, 
                                        Light* lights,
                                        size_t light_count,
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

    // obtain ray direction
    CU_Vector3f ray_dir = pixel_to_camera(u_x, u_y, 1.f, K);
    ray_dir.normalize();
    ray_dir = cam_rot*ray_dir;

    image[idx] = get_color(spheres, sphere_count, lights, light_count, camera_pos, ray_dir);
}

void raytrace_spheres(Sphere* spheres, size_t sphere_count, Light* lights, size_t light_count, int* visible, CU_Vector3f* vertices, CU_Vector3f* normals, CU_Vector3f* image, Camera* camera) {
    size_t vertex_count = camera->width * camera->height;

    Sphere* d_spheres;
    Light* d_lights;
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
    cudaMalloc((void**)&d_lights, light_count*sizeof(struct Light));
    // cudaMalloc((void**)&d_cam_rot, sizeof(CU_Matrix<3>));
    // cudaMalloc((void**)&d_cam_trans, sizeof(CU_Vector3f));
    // cudaMalloc((void**)&d_K, sizeof(CU_Matrix<3>));
    cudaMalloc((void**)&d_visible, vertex_count*sizeof(int));
    cudaMalloc((void**)&d_image, vertex_count*sizeof(CU_Vector3f));
    cudaMalloc((void**)&d_vertices, vertex_count*sizeof(CU_Vector3f));
    cudaMalloc((void**)&d_normals, vertex_count*sizeof(CU_Vector3f));

    cudaMemcpy(d_spheres, spheres, sphere_count*sizeof(struct Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights, light_count*sizeof(struct Light), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cam_rot, cam_rot, sizeof(CU_Matrix<3>), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cam_trans, cam_trans, sizeof(CU_Vector3f), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_K, camera-K, sizeof(CU_Matrix<3>), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid((camera->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (camera->height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    raytrace_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_spheres,
        sphere_count,
        d_lights,
        light_count,
        d_visible,
        d_vertices,
        d_normals,
        d_image,
        cam_rot,
        cam_trans,
        camera->K,
        camera->width,
        camera->height
    );
    cudaDeviceSynchronize();

    cudaMemcpy(visible, d_visible, vertex_count*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(image, d_image, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(vertices, d_vertices, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(normals, d_normals, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);

    cudaFree(d_spheres);
    cudaFree(d_lights);
    // cudaFree(d_cam_rot);
    // cudaFree(d_cam_trans);
    // cudaFree(d_K);
    cudaFree(d_visible);
    cudaFree(d_image);
    cudaFree(d_vertices);
    cudaFree(d_normals);
}