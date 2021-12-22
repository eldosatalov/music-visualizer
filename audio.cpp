/*******************************************************************************
*
*   Stuff for sound processing
*
*******************************************************************************/

#include <stdio.h>
#include <SDL2/SDL.h> // ans Uint8, etc.
//#include <complex>
#include <complex.h>
//#include <complex>
//#include <ccomplex>


//#include <ccomplex>
#include "audio.h"

//#include <tgmath.h>
//#include <complex.h>
#include <math.h>

//using namespace std;

#define BYTES_PER_SAMP 4        // assumes 16 bit 2 channel input

/******************************************************************************/
void MyAudioCallback(void* userdata, Uint8* stream, int streamLength)
{
  AudioData* audio = (AudioData*)userdata;

  if(audio->length == 0)
  {
    return;
  }

  Uint32 length = (Uint32)streamLength;
  length = (length > audio->length ? audio->length : length);

  SDL_memcpy(stream, audio->pos, length);

  audio->pos += length;
  audio->length -= length;

  // printf("reloading buffer\n");
	// 4096 samples per load (16 bit, 2 channel)
	// 44.1 samps per millisecond
	// about 93 milliseconds per load
}

/******************************************************************************/
int runAudio(GPU_Palette* P1, CPUAnimBitmap* A1, char* fileName)
{
	SDL_AudioSpec wavSpec;	// audio file header info (channels, samp rate, etc)
  Uint8* wavPtr;					// address of start of audio data
  Uint32 wavLength;				// length of music in file sample

	if(SDL_Init(SDL_INIT_AUDIO) < 0){
		printf("problem initializing SDL audio\n");
		return 1;
		}

  if(SDL_LoadWAV(fileName, &wavSpec, &wavPtr, &wavLength) == NULL){
    printf("error: file couldn't be found or isn't a proper .wav file\n");
    return 1;
  	}

	// if (wavSpec.freq != 44100){
	// 	printf(".wav frequency not 44100!\n");
	// 	return 1;
	// 	}

	// https://www.rubydoc.info/gems/sdl2_ffi/0.0.6/SDL2/Audio
	if (wavSpec.format != 0x8010){
		printf(".wav format is not S16LSB (signed 16 bit little endian)\n");
		return 1;
		}

	if (wavSpec.channels != 2){
		printf(".wav not 2 channel (stereo) file\n");
		return 1;
		}

	if (wavSpec.samples != 4096){
		printf(" SDL not using 4096 buffer size??!\n");
		return 1;
		}

  AudioData audio;
  audio.pos = wavPtr;				// address of where we are starting in the file
  audio.length = wavLength;		// how much data is left

  wavSpec.callback = MyAudioCallback;
  wavSpec.userdata = &audio;

  SDL_AudioDeviceID device = SDL_OpenAudioDevice(NULL, 0, &wavSpec, NULL, 0);

  if(device == 0){
    printf("error: audio device not found\n");
    return 1;
  }

  // 0 to play, 1 to pause
  SDL_PauseAudioDevice(device, 0);

  while(audio.length > 0)
  {
    // here is the main deal:
		myDraw(P1, A1, wavSpec.userdata);

    SDL_Delay(100);	// in ms
  }

  SDL_CloseAudioDevice(device);
  SDL_FreeWAV(wavPtr);
  SDL_Quit();
  return 0;
}



/******************************************************************************/
float getAmp(Uint8* wavPtr, int start, int end, int offset)
{
  // accumulate distance from zero across the frame
  float acc = 0;
  for (int samp = start; samp < end; samp++){ // e.g.  0 - 1023
  		acc += fabs(Get16bitAudioSample(wavPtr, (samp*BYTES_PER_SAMP)+offset));
  	}

  int divisor = (end-start);
  return (float) 1.0 * acc/divisor;
}



/******************************************************************************/
// the DFT is slower than the FFT, but more accurate
void getDFT(float* outBuff, Uint8* wavPtr, int theStart, int theEnd, int offset)
{
  // accumulate distance from zero across the frame

  // get buffer of sound to perform FFT on:
  int numSamps = theEnd-theStart; // e.g. 1024
  //complex double inBuffer[numSamps];
  //complex<double> inBuffer[numSamps];
  double _Complex inBuffer[numSamps];
  for (int samp = theStart; samp < theEnd; samp++){ // e.g.  0 - 1023
  	inBuffer[samp-theStart] =
            Get16bitAudioSample(wavPtr, (samp*BYTES_PER_SAMP)+offset);
  	}


  // little hack to make dft faster: 
  // compuite discrete fourier transform for the negative amplitudes only
  // negatives and positives look almost the same 

  // do DFT on input buffer, save in output buffer
//  complex double outBuffer[numSamps];
	for (int k = 0; k < numSamps / 2; k++) {  // For each output element
		//complex double sum = 0.0;
    //complex<double> sum = 0.0;
    double _Complex sum = 0.0;    
		for (int t = 0; t < numSamps / 2; t++) {  // For each input element
			double angle = 2 * M_PI * t * k / numSamps;
			sum += inBuffer[t] * cexp(-angle * I); // I is imaginary part
		}
		outBuff[k] = (fabs(creal(sum))); // convert from complex to float, rectify
	}
  // positives = negatives
  for (int k = 0; k < numSamps / 2; k++) {
    outBuff[numSamps - k - 1] = outBuff[k];
  }

#ifdef DETREND_AFTER_FFT
  // try to detrend amplitudes 
  int sub_bass = 2;
  int bass = 5;
  int low_midrange = 12;
  int midrange = 46;
  int upper_midrange = 93;
  int presence = 140;
  int brilliance = 512;

  float m_sb = 0;
  float m_b = 0;
  float m_lm = 0;
  float m_m = 0;
  float m_um = 0;
  float m_p = 0;
  float m_br = 0;

  for (int k = 0; k < numSamps; k++) {
    int f_k = k < numSamps - k - 1 ? k : numSamps - k - 1;
    if (f_k < sub_bass) {
      m_sb = m_sb > outBuff[k] ? m_sb : outBuff[k];
    } else if (f_k < bass) {
      m_b = m_b > outBuff[k] ? m_b : outBuff[k];
    } else if (f_k < low_midrange) {
      m_lm = m_lm > outBuff[k] ? m_lm : outBuff[k];
    } else if (f_k < midrange) {
      m_m = m_m > outBuff[k] ? m_m : outBuff[k];
    } else if (f_k < upper_midrange) {
      m_um = m_um > outBuff[k] ? m_um : outBuff[k];
    } else if (f_k < presence) {
      m_p = m_p > outBuff[k] ? m_p : outBuff[k];
    } else if (f_k < brilliance) {
      m_br = m_br > outBuff[k] ? m_br : outBuff[k];
    }
  }

  // slightly adjust
  m_sb *= 0.5;
  m_b *= 0.5;
  m_lm *= 0.6;
  m_m *= 0.8;
  m_um *= 1.3;
  m_p *= 1.3;
  m_br *= 1.3;

  for (int k = 0; k < numSamps; k++) {
    int f_k = k < numSamps - k - 1 ? k : numSamps - k - 1;
    if (f_k < sub_bass) {
      if (m_sb > 1) outBuff[k] /= m_sb;
    } else if (f_k < bass) {
      if (m_b > 1) outBuff[k] /= m_b;
    } else if (f_k < low_midrange) {
      if (m_lm > 1) outBuff[k] /= m_lm;
    } else if (f_k < midrange) {
      if (m_m > 1) outBuff[k] /= m_m;
    } else if (f_k < upper_midrange) {
      if (m_um > 1) outBuff[k] /= m_um;
    } else if (f_k < presence) {
      if (m_p > 1) outBuff[k] /= m_p;
    } else if (f_k < brilliance) {
      if (m_br > 1) outBuff[k] /= m_br;
    }
  }
#endif
//  float max = 0;
//  int location;
//  printf("outBuffer = [");
//	for (int k = 0; k < numSamps/2; k++) {
//    if (fabs(creal(outBuffer[k])) > max){
//      max = fabs(creal(outBuffer[k]));
//      location = k;
//    }
//    printf("%f, ", creal(outBuffer[k])); // print real part
//  }
//  printf("]\n");

//  int maxFreq = round(((location) * 44100)/1024);
//  printf("frequency = %d\n", maxFreq);


  // Q = 44.1kHz
  // Freqs for 1024pt DFT = [0, Q/1024, 2Q/1024, 3Q/1024 ... 1023Q/1024]
  // = [0, 43, 86, 129, .. 44,057Hz]

}

/******************************************************************************/
double Get16bitAudioSample(Uint8* bytebuffer, int samp)
{
    Uint16 val =  0x0;

//  assumes little endian
//    if(SDL_AUDIO_ISLITTLEENDIAN(format))
    val = (uint16_t)bytebuffer[samp] | ((uint16_t)bytebuffer[samp+1] << 8);
//    else
//        val = ((uint16_t)bytebuffer[0] << 8) | (uint16_t)bytebuffer[1];

//    if(SDL_AUDIO_ISSIGNED(format))
        return ((int16_t)val)/32768.0;

//    return val/65535.0;
}
