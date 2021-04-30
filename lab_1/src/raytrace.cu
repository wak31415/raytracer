#include "scene_objects.cuh"
#include "projection_helpers.cuh"
#include <iostream>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <ctime>
#include <math_constants.h>

#define GAMMA 2.2
#define MAX_RAY_DEPTH 7

#define INDIRECT_LIGHTING 1

__global__ void initialize_states(unsigned int seed, size_t width, curandState_t* states) {
    uint u_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint u_y = blockDim.y * blockIdx.y + threadIdx.y;

    uint idx = u_y * width + u_x;

    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                idx, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[idx]);
}

/* Get two independent normally distributed samples, store in x and y */
__device__ void boxMueller(curandState_t* states, unsigned int idx, float std, float* x, float* y) {
    float r1 = curand_uniform(&states[idx]);
    float r2 = curand_uniform(&states[idx]);

    *x = sqrt(-2 * log(r1)) * cos(2 * M_PI * r2) * std;
    *y = sqrt(-2 * log(r1)) * sin(2 * M_PI * r2) * std;
}

__device__ CU_Vector3f gamma_correct(CU_Vector3f color) {
    CU_Vector3f res;
    float exponent = 1.f/GAMMA;
    res[0] = powf(color[0], exponent);
    res[1] = powf(color[1], exponent);
    res[2] = powf(color[2], exponent);
    return res;
}

__device__ float get_distance(Sphere* spheres, 
                              size_t sphere_count,
                              CU_Vector3f ray,
                              CU_Vector3f start,
                              int* intersect_id)
{
    float min_dist = CUDART_INF_F;

    for(size_t i = 0; i < sphere_count; i++) {
        CU_Vector3f O_C = start - spheres[i].pos;

        float ray_dot_O_C = dot(ray, O_C);
        float O_C_norm = O_C.norm();
        float R = spheres[i].radius;
        float delta = ray_dot_O_C*ray_dot_O_C - O_C_norm*O_C_norm + R*R;

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
    return min_dist;
}

__device__ float get_distance(Triangle* triangles, 
                              size_t triangle_count,
                              CU_Vector3f ray,
                              CU_Vector3f start, 
                              int* intersect_id)
{
    float min_dist = CUDART_INF_F;

    for(size_t i = 0; i < triangle_count; i++) {
        Triangle T = triangles[i];
        CU_Vector3f e1 = T.B - T.A;
        CU_Vector3f e2 = T.C - T.A;
        CU_Vector3f A_O_cross_u = (T.A - start).cross(ray);
        CU_Vector3f N = e1.cross(e2);
        float rayDotN = dot(ray, N);

        float beta = dot(e2, A_O_cross_u) / rayDotN;
        float gamma = - dot(e1, A_O_cross_u) / rayDotN;

        float alpha = 1.f - beta - gamma;
        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            float t = dot(T.A - start, T.N) / rayDotN;
            if (*intersect_id < 0 || t < min_dist) {
                min_dist = t;
                *intersect_id = i;
            }
        }
    }
    return min_dist;
}

__device__ CU_Vector3f get_intersection(Sphere* spheres, 
                                        size_t sphere_count,
                                        Triangle* triangles,
                                        size_t triangle_count,
                                        CU_Vector3f ray,
                                        CU_Vector3f start, 
                                        int* sphere_id,
                                        int* triangle_id) 
{
    float min_dist = CUDART_INF_F;
    float min_dist_spheres = get_distance(spheres, sphere_count, ray, start, sphere_id);
    float min_dist_triangles = get_distance(triangles, triangle_count, ray, start, triangle_id);
    if(min_dist_spheres < min_dist_triangles) {
        min_dist = min_dist_spheres;
        *triangle_id = -1;
    } else {
        min_dist = min_dist_triangles;
        *sphere_id = -1;
    }
    if (__isinff(min_dist)) return CU_Vector3f(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F);
    return start + min_dist*ray;
}

__device__ bool is_visible(Sphere* spheres, size_t sphere_count, Triangle* triangles, size_t triangle_count, CU_Vector3f origin, CU_Vector3f target) {
    int sphere_id = -1;
    int triangle_id = -1;
    CU_Vector3f ray = target - origin;
    ray.normalize();
    CU_Vector3f P = get_intersection(spheres, sphere_count, triangles, triangle_count, ray, origin, &sphere_id, &triangle_id);
    if ((P - origin).norm() < (target - origin).norm()) return false;
    return true;
}

__device__ CU_Vector3f reflected_direction(CU_Vector3f ray, CU_Vector3f normal) {
    return ray - 2 * dot(ray, normal) * normal;
}

/**
 * Generates a random ray for indirect lighting
 **/
__device__ CU_Vector3f random_cos(curandState_t* states, CU_Vector3f normal, unsigned int idx) {
    float r1 = curand_uniform(&states[idx]);
    float r2 = curand_uniform(&states[idx]);

    float x = cosf(2*M_PI*r1)*sqrtf(1-r2);
    float y = sinf(2*M_PI*r1)*sqrtf(1-r2);
    float z = sqrtf(r2);

    // generate orthogonal vectors T1 and T2
    int k = normal.argmin_abs(); 
    int i = (int)fmod(k + 1.f, 3.f);
    int j = (int)fmod(k + 2.f, 3.f);

    // T1
    CU_Vector3f T1;
    T1[i] = normal[j];
    T1[j] = -normal[i];
    T1[k] = 0.f;
    T1.normalize();

    // T2
    CU_Vector3f T2 = normal.cross(T1);
    T2.normalize();

    return x*T1 + y*T2 + z*normal;
}

__device__ CU_Vector3f get_color(Sphere* spheres, 
                                 size_t sphere_count,
                                 Triangle* triangles,
                                 size_t triangle_count, 
                                 Light* lights,
                                 size_t light_count,
                                 CU_Vector3f start, 
                                 CU_Vector3f ray,
                                 bool* terminate_early,
                                 curandState_t* states,
                                 unsigned int idx)
{
    int depth = 0;
    CU_Vector3f L;
    CU_Vector3f albedo(1.f, 1.f, 1.f);

    while(depth < MAX_RAY_DEPTH) {
        int sphere_id = -1;
        int triangle_id = -1;
        CU_Vector3f P = get_intersection(spheres, sphere_count, triangles, triangle_count, ray, start, &sphere_id, &triangle_id);
        
        if(sphere_id >= 0 || triangle_id >= 0) {
            Material material;
            CU_Vector3f N;
            if(sphere_id >= 0) {
                material = spheres[sphere_id].material;
                N = P - spheres[sphere_id].pos;
                N.normalize();
            } else {
                material = triangles[triangle_id].material;
                N = triangles[triangle_id].N;
            }

            // Diffuse
            if(material.type == DIFFUSE) {
                // First surface we reach is diffuse, single ray is sufficient
                if(!INDIRECT_LIGHTING) {
                    if(depth == 0) *terminate_early = true;
                }

                // Normalized vector point --> light
                CU_Vector3f S_P = lights[0].pos - P;
                float d = S_P.norm();
                CU_Vector3f w_i = 1.f/d * S_P;

                float N_wi_dot = max(dot(N, w_i), 0.f);

                // check if the light is visible from P
                bool P_visible = is_visible(spheres, sphere_count, triangles, triangle_count, P+0.01*N, lights[0].pos);

                CU_Vector3f direct = lights[0].I / (4*M_PI*M_PI*d*d) * material.color * P_visible * N_wi_dot;
                
                L += albedo * direct;

                if(!INDIRECT_LIGHTING) return L;

                albedo *= material.color;

                start = P + 0.01*N;
                ray = random_cos(states, N, idx);
            }

            // Mirror
            else if(material.type == MIRROR) {
                CU_Vector3f reflected_ray = reflected_direction(ray, N);
                start = P + 0.01*N;
                ray = reflected_ray;
            }

            // Glass 
            else if(material.type == GLASS) {
                float ro = material.ro;
                float ri = material.ri;

                float wi_N_dot = dot(ray, N);

                // Fresnel
                float k0 = (ro - ri)*(ro - ri) / ((ro + ri)*(ro + ri));
                float R = k0 + (1.f-k0)*powf(1-abs(wi_N_dot), 5.f);

                // generate random numer
                float r = curand_uniform(&states[idx]);

                if(r < R) {
                    CU_Vector3f reflected_ray = reflected_direction(ray, N);
                    start = P+0.01*N;
                    ray = reflected_ray;

                } else {
                    CU_Vector3f wt_T;
                    CU_Vector3f wt_N;

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
            }
        }
        else {
            return L;
        }
        depth = depth + 1;
    }
    return L;
}

__global__ void raytrace_spheres_kernel(Sphere* spheres, 
                                        size_t sphere_count, 
                                        Triangle* triangles,
                                        size_t triangle_count,
                                        Light* lights,
                                        size_t light_count,
                                        CU_Vector3f* image, 
                                        CU_Matrix<3> cam_rot,
                                        CU_Vector3f camera_pos,
                                        CU_Matrix<3> K,
                                        uint width,
                                        uint height,
                                        uint num_rays,
                                        curandState_t* states,
                                        volatile int* progress) 
{
    uint u_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint u_y = blockDim.y * blockIdx.y + threadIdx.y;

    uint idx = u_y * width + u_x;

    if (u_x >= width || u_y >= height) 
        return;

    bool terminate_early = false;
    CU_Vector3f color(0.f, 0.f, 0.f);

    for(int i = 0; i < num_rays; i++) {
        float dx=1.f, dy=1.f;

        // randomizing ray direction for anti-aliasing
        while(abs(dx) > 0.5f || abs(dy) > 0.5f)
            boxMueller(states, idx, 1.f, &dx, &dy);

        if(abs(dx) <= 0.5f && abs(dy) <= 0.5f) {
            // obtain ray direction
            // if(idx==0) printf("Ray %d\n", i);
            CU_Vector3f ray_dir = pixel_to_camera(u_x+0.5f+dx, u_y+0.5f+dx, 1.f, K);
            ray_dir.normalize();
            ray_dir = cam_rot*ray_dir;

            color += get_color(spheres, sphere_count, triangles, triangle_count, lights, light_count, camera_pos, ray_dir, &terminate_early, states, idx);
        }
    }
    image[idx] = gamma_correct((1.f/num_rays) * color);

    // Update progress
    if (!(threadIdx.x || threadIdx.y)){
        atomicAdd((int *)progress, 1);
        __threadfence_system();
    }
}

void raytrace_spheres(Sphere* spheres, 
                      size_t sphere_count, 
                      Triangle* triangles, 
                      size_t triangle_count, 
                      Light* lights, 
                      size_t light_count, 
                      CU_Vector3f* image, 
                      Camera* camera) 
{
    size_t vertex_count = camera->width * camera->height;

    curandState_t* d_states;
    Sphere* d_spheres;
    Triangle* d_triangles;
    Light* d_lights;
    CU_Matrix<4> d_cam_rot;
    CU_Vector3f d_cam_trans;
    CU_Matrix<3> d_K;
    CU_Vector3f* d_image;

    CU_Matrix<3> cam_rot = camera->E.get_rotation();
    CU_Vector3f cam_trans = camera->E.get_translation();

    cudaMalloc((void**)&d_states, vertex_count*sizeof(curandState_t));
    cudaMalloc((void**)&d_spheres, sphere_count*sizeof(Sphere));
    cudaMalloc((void**)&d_triangles, triangle_count*sizeof(Triangle));
    cudaMalloc((void**)&d_lights, light_count*sizeof(struct Light));
    cudaMalloc((void**)&d_image, vertex_count*sizeof(CU_Vector3f));

    cudaMemcpy(d_spheres, spheres, sphere_count*sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, triangles, triangle_count*sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights, light_count*sizeof(struct Light), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 16);
    dim3 blocksPerGrid((camera->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (camera->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    volatile int *d_data, *h_data;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void **)&h_data, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer((int **)&d_data, (int *)h_data, 0);
    *h_data = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);

    clock_t t = std::clock();

    initialize_states<<<blocksPerGrid, threadsPerBlock>>>(time(0), camera->width, d_states);
    
    cudaDeviceSynchronize();

    raytrace_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_spheres,
        sphere_count,
        d_triangles,
        triangle_count,
        d_lights,
        light_count,
        d_image,
        cam_rot,
        cam_trans,
        camera->K,
        camera->width,
        camera->height,
        camera->num_rays,
        d_states,
        d_data
    );
    cudaEventRecord(stop);

    unsigned int num_blocks = blocksPerGrid.x*blocksPerGrid.y;
    float my_progress = 0.0f;
    do{
        cudaEventQuery(stop);  // may help WDDM scenario
        int value1 = *h_data;
        float kern_progress = (float)value1/(float)num_blocks;
        if ((kern_progress - my_progress)> 0.02f) {
            float time_passed = (float)(std::clock() - t)/static_cast<float>(CLOCKS_PER_SEC);
            float eta = time_passed / kern_progress - time_passed;
            std::cout << "\rProgress: [";
            for(size_t i = 0; i < 20; i++) {
                if(i/20.f < my_progress) {
                    std::cout << "#";
                } else {
                    std::cout << " ";
                }
            }
            char s[10];
            memset((void*)s, 0, 10*sizeof(char));
            std::snprintf(s, 10*sizeof(char), "%.2f", eta);
            std::cout << "] - "<< static_cast<int>(kern_progress*100) << "% \t Remaining: " << (char*)s << "s" << std::flush;
            // fflush(stdout);
            my_progress = kern_progress;
        }
    }
    while (my_progress < 0.98f);
    printf("\n");

    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaDeviceSynchronize();
    printf("Finished raytracing in %.3f seconds.\n", (double)(std::clock() - t)/CLOCKS_PER_SEC);//et/1000.f);

    cudaMemcpy(image, d_image, vertex_count*sizeof(CU_Vector3f), cudaMemcpyDeviceToHost);

    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_lights);
    cudaFree(d_image);
}