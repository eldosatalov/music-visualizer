#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // more rand stuff
#include <cuda_texture_types.h>

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct GPU_Palette{

    unsigned int palette_width;
    unsigned int palette_height;
    unsigned long num_pixels;

    dim3 gThreads;
    dim3 gBlocks;

//    float* gray;
    float* red;
    float* green;
    float* blue;
    float* dft;
    curandState* rand;
};

GPU_Palette initGPUPalette(unsigned int, unsigned int);

int updatePalette(GPU_Palette*, float, float);
void freeGPUPalette(GPU_Palette*);

// kernel calls:
//__global__ void updateGrays(float* gray);
__global__ void updateReds(float* red, curandState* gRand, unsigned long);
__global__ void updateGreens(float* green, unsigned long, float* dft);
__global__ void updateGreensInCircle(float* green, unsigned long, float* dft);
__global__ void updateBlues(float* blue, unsigned long, float,float);
__global__ void updateBluesInCircle(float* blue, unsigned long, float* dft);
__global__ void setup_rands(curandState* state, unsigned long seed, unsigned long);


#endif  // GPULib
