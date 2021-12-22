/**************************************************************************
*
*     Music Visualizer Base Code
*
**************************************************************************/

#include <cuda.h>
#include <curand.h>               // includes random num stuff
#include <curand_kernel.h>       	// more rand stuff
#include <cuda_texture_types.h>   // need this for

#include <stdio.h>
#include "gpu_main.h"

// define texture memory
// texture<float, 2> texGray;
texture<float, 2> texBlue;
// texture<float, 2> texGreen;
// texture<float, 2> texBlue;

//unsigned long num_pixels;

/******************************************************************************/
GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
  GPU_Palette X;

  X.gThreads.x = 32;  // 32 x 32 = 1024 threads per block
  X.gThreads.y = 32;
  X.gThreads.z = 1;
  X.gBlocks.x = ceil(imageWidth/32);  // however many blocks needed for image
  X.gBlocks.y = ceil(imageWidth/32);
  X.gBlocks.z = 1;

  X.palette_width = imageWidth;       // save this info
  X.palette_height = imageHeight;
  X.num_pixels = imageWidth * imageHeight;

  // allocate memory on GPU
  cudaError_t err;
  err = cudaMalloc((void**) &X.red, X.num_pixels * sizeof(float));
  if(err != cudaSuccess){
    printf("cuda error allocating red = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.green, X.num_pixels * sizeof(float)); // g
  if(err != cudaSuccess){
    printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.blue, X.num_pixels * sizeof(float));  // b
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.rand, X.num_pixels * sizeof(curandState));
  if(err != cudaSuccess){
    printf("cuda error allocating rands = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  // dft size = 1024
  err = cudaMalloc((void**) &X.ldft, 1024 * sizeof(float));
  if(err != cudaSuccess){
    printf("cuda error allocating dft = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.rdft, 1024 * sizeof(float));
  if(err != cudaSuccess){
    printf("cuda error allocating dft = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }


  // init rand seeds on the gpu
  setup_rands <<< X.gBlocks, X.gThreads >>> (X.rand, time(NULL), X.num_pixels);

  // create texture memory and bind
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  unsigned int pitch = sizeof(float) * imageWidth;
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, imageWidth, imageHeight, pitch);

  return X;
}

/******************************************************************************/
void freeGPUPalette(GPU_Palette* P)
{
  // free texture memory
  cudaUnbindTexture(texBlue); // this is bound to black and white

  // free gpu memory
//  cudaFree(P->gray);
  cudaFree(P->red);
  cudaFree(P->green);
  cudaFree(P->blue);
  cudaFree(P->ldft);
  cudaFree(P->rdft);
  cudaFree(P->rand);
}


/******************************************************************************/
int updatePalette(GPU_Palette* P, float lAmp, float rAmp)
{

  // drop your reds, drop your greens and blues :)
  // updateReds <<< P->gBlocks, P->gThreads >>> (P->red, P->rand, P->num_pixels);
  //updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, P->num_pixels, P->dft);
  updateBluesInCircle <<< P->gBlocks, P->gThreads >>> (P->blue, P->num_pixels, P->ldft);
	updateGreensInCircle <<< P->gBlocks, P->gThreads >>> (P->green, P->num_pixels, P->rdft);
  // updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, P->num_pixels, lAmp, rAmp);
  updateRandCircle <<< P->gBlocks, P->gThreads >>> (P->red, P->green, P->blue, P->num_pixels, lAmp, rAmp);

  return 0;
}

/******************************************************************************/
__global__ void updateRandCircle(float* red, float* green, float* blue, unsigned long numPixels, float lAmp, float rAmp)
{
  // assuming 1024w x 512h palette
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int center_x = 512;
  int center_y = 256;
  int rad = 45;

  float dist = hypot(center_x - x, center_y - y);

  if(vecIdx < numPixels && dist < rad){
    red[vecIdx] = 0.3;
    green[vecIdx] = rAmp * 3;
    blue[vecIdx] = lAmp  *3;
  }
}

/******************************************************************************/
__global__ void updateReds(float* red, curandState* gRand, unsigned long numPixels)
{
  // assuming 1024w x 512h palette
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if(vecIdx < numPixels){

    // generate noise
    curandState localState = gRand[vecIdx];
    float theRand = curand_uniform(&localState); // value between 0-1
  //  float theRand = curand_poisson(&localState, .5);
    gRand[vecIdx] = localState;

    // sparkle the reds:
    if(theRand > .999) red[vecIdx] = red[vecIdx] *.9;
    else if(theRand < .001) red[vecIdx] = (1.0-red[vecIdx]);
    }
}

/******************************************************************************/
__global__ void updateGreens(float* green, unsigned long numPixels, float* dft)
{
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  // assuming 1024w x 512h palette
  if(vecIdx < numPixels){ // don't compute pixels out of range
    float theVal = dft[x]/800; // 390 - play with this value
    if(theVal > 1.0) theVal = 1.0;  //error check in range
    if(theVal < 0.0) theVal = 0.0;

    if(y < floor(theVal*512)) green[vecIdx] = 1.0;
    else green[vecIdx] = 0;
  }
}

#include <math.h>

/******************************************************************************/
__global__ void updateGreensInCircle(float* green, unsigned long numPixels, float* dft)
{
  float pi = 3.14159265359;
  int center_x = 512;
  int center_y = 256;

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  float a_per_ray = pi / 1024;
  if (vecIdx < numPixels && x >= center_x) {
    float angle = atan2(y - center_y, x - center_x) + pi + pi / 2;
    int fr = angle / a_per_ray;
    fr = fr % 1024;

    float theVal = dft[fr];
    float rad = 250;
    float in_r = 50;
    float dist = hypot(x - center_x, y - center_y);
    if(theVal > 1.0) theVal = 1.0;  //error check in range
    if(theVal < 0.0) theVal = 0.0;
    if (dist <= rad * theVal && dist > in_r) {
      green[vecIdx] = 1.0;
    }
    else if (dist > in_r) {
      green[vecIdx] = 0;
    }
  }

}

/******************************************************************************/
__global__ void updateBluesInCircle(float* blue, unsigned long numPixels, float* dft)
{
  float pi = 3.14159265359;
  int center_x = 512;
  int center_y = 256;
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  float a_per_ray = pi / 1024;
  if (vecIdx < numPixels && x < center_x) {
    float angle = atan2(y - center_y, x - center_x) + pi + 3 * pi / 2;
    int fr = angle / a_per_ray;
    fr %= 1024;
    fr = 1023 - fr;
    float theVal = dft[fr];
    float rad = 250;
    float in_r = 50;
    float dist = hypot(x - center_x, y - center_y);
    if(theVal > 1.0) theVal = 1.0;  //error check in range
    if(theVal < 0.0) theVal = 0.0;
    if (dist <= rad * theVal && dist > in_r) {
      blue[vecIdx] = 1.0;
    }
    else if (dist > in_r) {
      blue[vecIdx] = 0;
    }
  }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, unsigned long numPixels, float lAmp, float rAmp)
{

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if(vecIdx < numPixels){
    if (x < 512) blue[vecIdx] = lAmp;
    else blue[vecIdx] = rAmp;
  }
}

/******************************************************************************/
__global__ void setup_rands(curandState* state, unsigned long seed, unsigned long numPixels)
{

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if(vecIdx < numPixels) curand_init(seed, vecIdx, 0, &state[vecIdx]);
}

/******************************************************************************/
