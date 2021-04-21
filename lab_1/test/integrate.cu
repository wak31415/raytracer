#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define N_SAMPLES 10000

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
}

__device__ double gaussian(double sigma, double x, double y, double z) {
    // return expf(- (x*x + y*y + z*z) / (2*sigma*sigma));
    return 1.f/pow(sigma*sqrt(2*M_PI), 3.f) * exp(- (x*x + y*y + z*z) / (2*sigma*sigma));
}


/* Get two independent normally distributed samples, store in x and y */
__device__ void boxMueller(curandState_t* states, double std, double* x, double* y) {
    double r1 = curand_uniform(&states[blockIdx.x]);
    double r2 = curand_uniform(&states[blockIdx.x]);

    *x = sqrt(-2 * log(r1)) * cos(2 * M_PI * r2) * std;
    *y = sqrt(-2 * log(r1)) * sin(2 * M_PI * r2) * std;
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void monte_carlo_integration(curandState_t* states, double* numbers) {
    unsigned int idx = blockIdx.x;

    double x1, y1, z1, 
           x2, y2, z2;
    boxMueller(states, 1.f, &x1, &x2);
    boxMueller(states, 1.f, &y1, &y2);
    boxMueller(states, 1.f, &z1, &z2);

    if(x1 < -M_PI_2 || x1 > M_PI_2 ||
       y1 < -M_PI_2 || y1 > M_PI_2 ||
       z1 < -M_PI_2 || z1 > M_PI_2) 
    {
        numbers[2*idx] = 0.f;
    } else {
        double f1 = cos(x1*y1*z1);
        double pdf1 = gaussian(1.f, x1, y1, z1);
        numbers[2*idx] = f1 / pdf1;
    }

    if(x2 < -M_PI_2 || x2 > M_PI_2 ||
       y2 < -M_PI_2 || y2 > M_PI_2 ||
       z2 < -M_PI_2 || z2 > M_PI_2) 
    {
        numbers[2*idx+1] = 0.f;
    } else {
        double f2 = cos(x2*y2*z2);
        double pdf2 = gaussian(1.f, x2, y2, z2);
        numbers[2*idx+1] = f2 / pdf2;
    }
}

int main() {
    curandState_t* states;
    cudaMalloc((void**) &states, N_SAMPLES * sizeof(curandState_t));

    /* invoke the GPU to initialize all of the random states */
    init<<<N_SAMPLES, 1>>>(time(0), states);

    /* allocate an array of unsigned ints on the CPU and GPU */
    double* cpu_nums = (double*)malloc(N_SAMPLES*sizeof(double));
    double* gpu_nums;

    cudaMalloc((void**) &gpu_nums, N_SAMPLES * sizeof(double));

    monte_carlo_integration<<<N_SAMPLES/2, 1>>>(states, gpu_nums);

    cudaMemcpy(cpu_nums, gpu_nums, N_SAMPLES * sizeof(double), cudaMemcpyDeviceToHost);

    double s = 0.f;
    for (int i = 0; i < N_SAMPLES; i++) {
        s += cpu_nums[i];
    }

    printf("F = %.4f\n", s / N_SAMPLES);

    /* free the memory we allocated for the states and numbers */
    cudaFree(states);
    cudaFree(gpu_nums);
    free(cpu_nums);

    return 0;
}