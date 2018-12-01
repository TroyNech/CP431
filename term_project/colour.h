/*
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: November 16, 2018
*
* Note: Valid colour values are [0,255]
*/

typedef struct colour {
    int red, green, blue;
} colour;

#define COLOUR1 {255, 0, 0} //red
#define COLOUR2 {255, 128, 0} //orange
#define COLOUR3 {255, 255, 0} //yellow
#define COLOUR4 {0, 255, 0} //green
#define COLOUR5 {0, 255, 255} //cyan
#define COLOUR6 {0, 0, 255} //blue
#define COLOUR7 {127, 0, 255} //purple
#define COLOUR8 {255, 0, 255} //pink
#define COLOUR9 {128, 128, 128} //gray
#define COLOUR10 {255, 255, 255} //white

#define COLOURS_SIZE 10

const colour COLOURS[10] = {COLOUR1, COLOUR2, COLOUR3, COLOUR4, COLOUR5, COLOUR6, COLOUR7, COLOUR8, COLOUR9, COLOUR10}