#ifndef AUDIOLib
#define AUDIOLib

#include "gpu_main.h"
#include "animate.h"

struct AudioData
{
  Uint8* pos;
  Uint32 length;
};

// This is the main function to work with - called once every 100ms or so..
void myDraw(GPU_Palette* P1, CPUAnimBitmap* A1, void* theAudio);

int runAudio(GPU_Palette* P1, CPUAnimBitmap* A1, char* fileName);
void MyAudioCallback(void* userdata, Uint8* stream, int streamLength);

float getAmp(Uint8* wavPtr, int start, int end, int offset);
void getDFT(float*, Uint8* wavPtr, int theStart, int theEnd, int offset);

double Get16bitAudioSample(Uint8* bytebuffer, int);

// NEED TO WRITE SOME MORE SOUND HANDLING FUNCTIONS..
//float getFreq(Uint8* wavPtr, int start, int end, int offset);


#endif // AUDIOliB
