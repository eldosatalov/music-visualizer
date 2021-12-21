/*******************************************************************************
*
*                        Header info goes here
*
*******************************************************************************/
#ifndef ANIMLib
#define ANIMLib

#include <GL/freeglut.h>
#include <GL/glext.h>
#include <GL/glx.h>

#include "gpu_main.h" // has GPU_PALETTE

struct CPUAnimBitmap {
    unsigned char* pixels;
    int width;
    int height;
    void (* clickDrag)(void*, int, int, int, int);
    int dragStartX, dragStartY;

    GPU_Palette* thePalette;
    unsigned char* dev_bitmap;

    CPUAnimBitmap(GPU_Palette*);

    ~CPUAnimBitmap();

    unsigned char* get_ptr(void) const { return pixels; }

    long image_size(void) const { return width * height * 4; }

    void click_drag(void (* f)(void*, int, int, int, int));

    //void anim_and_exit(void (* f)(void*, int), void(* e)(void*));
    void anim_and_exit(void (* f)(void*), void(* e)(void*));

    // static method used for glut callbacks
    static CPUAnimBitmap** get_bitmap_ptr(void);

    // static method used for glut callbacks
    static void mouse_func(int button, int state, int mx, int my);

    // static method used for glut callbacks
    static void idle_func(void);

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y);
    // static method used for glut callbacks
    static void Draw(void);

    void initAnimation();
    void drawPalette(void);
};


__global__ void drawGray(unsigned char* optr, const float* gray);
__global__ void drawColor(unsigned char* optr,
                          const float* red,
                          const float* green,
                          const float* blue);

#endif
