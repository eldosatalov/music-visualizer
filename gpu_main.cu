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
  cudaMalloc((void**) &X.green, X.num_pixels * sizeof(float)); // g
  if(err != cudaSuccess){
    printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  cudaMalloc((void**) &X.blue, X.num_pixels * sizeof(float));  // b
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  cudaMalloc((void**) &X.rand, X.num_pixels * sizeof(curandState));
  if(err != cudaSuccess){
    printf("cuda error allocating rands = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  // dft size = 1024
  cudaMalloc((void**) &X.dft, 1024 * sizeof(float));
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
  cudaFree(P->dft);
  cudaFree(P->rand);
}


/******************************************************************************/
int updatePalette(GPU_Palette* P, float lAmp, float rAmp)
{

  // drop your reds, drop your greens and blues :)
  updateReds <<< P->gBlocks, P->gThreads >>> (P->red, P->rand, P->num_pixels);
  updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, P->num_pixels, P->dft);
	updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, P->num_pixels, lAmp, rAmp);

  return 0;
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
    float theVal = dft[x]/390.0; // 390 - play with this value
    if(theVal > 1.0) theVal = 1.0;  //error check in range
    if(theVal < 0.0) theVal = 0.0;

    if(y < floor(theVal*512)) green[vecIdx] = 1.0;
    else green[vecIdx] = 0;
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
