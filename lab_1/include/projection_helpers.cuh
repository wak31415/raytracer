// --------------- WORLD -> OTHER --------------- //

/**
* @param p_world  point in world space
* @param E        Camera extrinsic matrix
**/
__host__ __device__ CU_Vector<3> world_to_camera(CU_Vector<4> p_world,
                                                 CU_Matrix<4> E)
{
    CU_Vector<4> p_camera = E*p_world;
    return CU_Vector3f(p_camera[0], p_camera[1], p_camera[2]);
}

/**
* @param p_world  point in world space
* @param E        Camera extrinsic matrix
**/
__host__ __device__ CU_Vector<3> world_to_camera(CU_Vector<3> p_world,
                                                 CU_Matrix<4> E) 
{   
    float tmp[4] = {p_world[0], p_world[1], p_world[2], 1.f};
    CU_Vector<4> p_world_4f(tmp);
    return world_to_camera(p_world_4f, E);
}

/**
* @param p_world  point in world space
* @param E        Camera extrinsic matrix
* @param K        Camera intrinsic matrix
**/
template <size_t VecSize>
__host__ __device__ CU_Vector<3> world_to_image(CU_Vector<VecSize> p_world, 
                                                CU_Matrix<4> E, 
                                                CU_Matrix<3> K)
{
    CU_Vector3f p_camera = world_to_camera(p_world, E);

    // p_img = K * p_camera
    CU_Vector3f p_img;
    p_img[0] = K[0*3 + 0]*p_camera[0] + K[0*3 + 2]*p_camera[2];
    p_img[1] = K[1*3 + 1]*p_camera[1] + K[1*3 + 2]*p_camera[2];
    p_img[2] = p_camera[2];
    return p_img;
}

/**
* @param p_world  point in world space
* @param E        Camera extrinsic matrix
* @param K        Camera intrinsic matrix
**/
template <size_t VecSize>
__host__ __device__ CU_Vector<2> world_to_pixel(CU_Vector<VecSize> p_world, 
                                                CU_Matrix<4> E, 
                                                CU_Matrix<3> K)
{
    CU_Vector3f p_img = world_to_image(p_world, E, K);
    float tmp[2] = {p_img[0]/p_img[2], p_img[1]/p_img[2]};
    return CU_Vector<2>(tmp);
}


// --------------- PIXEL -> OTHER --------------- //

/**
* @param pixel      point in pixel space
* @param depth      depth value (from camera)
* @param K          Camera intrinsic matrix
**/
__host__ __device__ CU_Vector3f pixel_to_camera(float u_x, float u_y, float depth, CU_Matrix<3> K)
{
    // p_camera = K_inv * p_img
    const float x = (u_x - K[0*3 + 2]) / K[0*3 + 0];
    const float y = (u_y - K[1*3 + 2]) / K[1*3 + 1];
    return CU_Vector3f(x, y, depth);
}


/**
* @param p_camera   point in camera space
* @param E_inv      camera extrinsic inverse matrix
**/
__host__ __device__ CU_Vector3f camera_to_world(CU_Vector3f p_camera, CU_Matrix<4> E_inv)
{
    float tmp[4] = {p_camera[0], p_camera[1], p_camera[2], 1.f};
    CU_Vector<4> p_camera_4f(tmp);
    CU_Vector<4> p_world = E_inv * p_camera_4f;
    return CU_Vector3f(p_world[0], p_world[1], p_world[2]);
}


/**
* @param pixel      point in pixel space
* @param depth      depth value (from camera)
* @param K          Camera intrinsic matrix
* @param E_inv      camera extrinsic inverse matrix
**/
__host__ __device__ CU_Vector3f pixel_to_world(uint u_x, uint u_y, float depth, CU_Matrix<3> K, CU_Matrix<4> E_inv)
{
    CU_Vector3f p_camera = pixel_to_camera(u_x, u_y, depth, K);
    CU_Vector3f p_world = camera_to_world(p_camera, E_inv);
    return p_world;
}