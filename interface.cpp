/*******************************************************************************
*
*   Music visualizer base code by Dr. Michael Brady
*
*******************************************************************************/
#include <stdio.h>
#include <cstdlib>		// need this?
#include <string.h>		// need this?
#include <time.h>
#include <unistd.h> 	// includes usleep
#include <SDL2/SDL.h>	// for sound processing..

#include "interface.h"
#include "gpu_main.h"
#include "animate.h"
#include "crack.h"
#include "audio.h"


int VERBOSE = 1; 		// only used for interface
int RUNMODE = 1;		// only used for interface


/******************************************************************************/
int main(int argc, char *argv[]){

	unsigned char ch;

  // -- get parameters that differ from defaults from command line:
  if(argc<2){usage(); return 1;} // must be at least one arg (fileName)
	while((ch = crack(argc, argv, "r|v|f|", 0)) != NULL) {
	  switch(ch){
    	case 'r' : RUNMODE = atoi(arg_option); break;
      case 'v' : VERBOSE = atoi(arg_option); break;
      default  : usage(); return(0);
    }
  }

  // error handling & runmode for filename.wav or filename.bmp
	char* fileName = argv[arg_index];

  printf("\n\nlaunching GPU...\n\n");
	GPU_Palette P1;
	P1 = openPalette(1024, 512); // width, height
	CPUAnimBitmap animation(&P1);
	cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

  switch(RUNMODE){
			case 0:
				if (VERBOSE) printf("\n -- printing machine info -- \n");
				runMode0(); // sanity check: print info about GPU
				break;
      case 1:
        if (VERBOSE) printf("\n -- running music visualizer -- \n");
				runAudio(&P1, &animation, fileName);
        break;

      default: printf("no valid run mode selected\n");
  }

return 0;
}

/******************************************************************************/
// print information about GPU
int runMode0(void)
{
  cudaError_t err;
//  err = cudaDeviceReset();
//  if(err != cudaSuccess){
//    printf("problem resetting device = %s\n", cudaGetErrorString(err));
//    return 1;
//    }

  // GET INFORMATION ABOUT GPU DEVICE(S)
  cudaDeviceProp prop;
  int count;
  err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    printf("problem getting device count = %s\n", cudaGetErrorString(err));
    return 1;
    }
  printf("number of GPU devices: %d\n\n", count);

  for (int i = 0; i< count; i++){
    printf("************ GPU Device: %d ************\n\n", i);
    err = cudaGetDeviceProperties(&prop, i);
    if(err != cudaSuccess){
      printf("problem getting device properties = %s\n", cudaGetErrorString(err));
      return 1;
      }

    printf("\tName: %s\n", prop.name);
    printf( "\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf( "\tClock rate: %d\n", prop.clockRate );
    printf( "\tDevice copy overlap: " );
      if (prop.deviceOverlap)
        printf( "Enabled\n" );
      else
        printf( "Disabled\n" );
    printf( "\tKernel execition timeout: " );
      if (prop.kernelExecTimeoutEnabled)
        printf( "Enabled\n" );
      else
        printf( "Disabled\n" );
//    printf( "--- Memory Information for device %d ---\n", i );
    printf("\tTotal global mem: %ld\n", prop.totalGlobalMem );
    printf("\tTotal constant Mem: %ld\n", prop.totalConstMem );
    printf("\tMax mem pitch: %ld\n", prop.memPitch );
    printf( "\tTexture Alignment: %ld\n", prop.textureAlignment );
    printf("\n");
    printf( "\tMultiprocessor count: %d\n", prop.multiProcessorCount );
    printf( "\tShared mem per processor: %ld\n", prop.sharedMemPerBlock );
    printf( "\tRegisters per processor: %d\n", prop.regsPerBlock );
    printf( "\tThreads in warp: %d\n", prop.warpSize );
    printf( "\tMax threads per block: %d\n", prop.maxThreadsPerBlock );
    printf( "\tMax block dimensions: (%d, %d, %d)\n",
                  prop.maxThreadsDim[0],
                  prop.maxThreadsDim[1],
                  prop.maxThreadsDim[2]);
    printf( "\tMax grid dimensions: (%d, %d, %d)\n",
                  prop.maxGridSize[0],
                  prop.maxGridSize[1],
                  prop.maxGridSize[2]);
    printf("\n");
  }

  return 0;
}

/******************************************************************************/
GPU_Palette openPalette(int theWidth, int theHeight)
{
	unsigned long theSize = theWidth * theHeight;

	unsigned long memSize = theSize * sizeof(float);
	//float* graymap = (float*) malloc(P1->gSize);
	float* redmap = (float*) malloc(memSize);
	float* greenmap = (float*) malloc(memSize);
	float* bluemap = (float*) malloc(memSize);

	for(int i = 0; i < theSize; i++)
	{
  	bluemap[i]    = 0;
  	greenmap[i]  = 0;
  	redmap[i]   = 0;
	}

	GPU_Palette P1 = initGPUPalette(theWidth, theHeight);

	//cudaMemcpy(P1.gray, graymap, memSize, cH2D);
	cudaMemcpy(P1.red, redmap, memSize, cH2D);
	cudaMemcpy(P1.green, greenmap, memSize, cH2D);
	cudaMemcpy(P1.blue, bluemap, memSize, cH2D);

	//free(graymap);
	free(redmap);
	free(greenmap);
	free(bluemap);

	return P1;
}

/******************************************************************************/
// This is the core function for drawing audio. The pointer to the current
// location of the audio data (audio->pos) is updated about once every 100 ms
// and is advanced 4096 samples. myDraw() is called from where this happens
// and thus myDraw() is called about once every 100ms, or 10x per second.
//
// So, e.g., if you want to update the GPU palette 40x per second (40Hz),
// divide the processing into 4 windows at 1024 samples per window.
// 1024 is also a decent FFT size, but it's good to do overlapping windows..
//
// Note: Hollywood movies are generally 24 frames per second, but the
// new Hobbit movie is 48 frames per second.
//
// function prototype is included in audio.h

void myDraw(GPU_Palette* P1, CPUAnimBitmap* A1, void* theAudio){

	AudioData* audio = (AudioData*)theAudio;

	int numSampsPerFrame = 1024; // a sample is 4 bytes (16 bits x 2 channels)
	int numFrames = 4;

	int frameStart;
	int frameEnd;
	float leftAmp = 0;
	float rightAmp = 0;
	float dftBuff[numSampsPerFrame];
	for (int i = 0; i < numFrames; i++){
//		printf("frame = %d", i);
		frameStart = (i * numSampsPerFrame);
		frameEnd = ((i+1) * numSampsPerFrame);
		leftAmp = getAmp(audio->pos, frameStart, frameEnd, 0);
		rightAmp = getAmp(audio->pos, frameStart, frameEnd, 2);

		// get DFT on right channel
		getDFT(dftBuff, audio->pos, frameStart, frameEnd, 0);	// 2nd

		cudaMemcpy(P1->ldft, dftBuff, 1024 * sizeof(float), cH2D);

		//get DFT on left channel
		getDFT(dftBuff, audio->pos, frameStart, frameEnd, 2);  // 1st

    cudaMemcpy(P1->rdft, dftBuff, 1024 * sizeof(float), cH2D);

		int err = updatePalette(P1, leftAmp, rightAmp);
    A1->drawPalette();


		usleep(23000); // sleep 23 milliseconds
		}
}

/******************************************************************************/
int usage()
{
	printf("USAGE:\n");
	printf("-r[val] filename\n\n");
  printf("e.g.> ex2 -r1 -v1 filename.wav\n");
  printf("v  verbose mode (0:none, 1:normal, 2:params\n");
  printf("r  run mode 0:GPU info, 1:music visualizer\n\n");
	printf("note: be sure .wav file is 16 bit, 2 channel, 44.1kHz\n");

  return(0);
}

/******************************************************************************/
