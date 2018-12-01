/*
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: November 31, 2018
*
* Usage: julia_set c_real c_imag
* Description:
*   Creates an image of the Julia set for z^2 + c using 3000x3000 points from (-2,-2) to (2,2)
* Compile like: mpicc julia_set.c -o julia_set -std=c99 -lm -lglut -lGL -lGLU
*/

#include "bitmap.h"
#include <complex.h>
#include "colour.h"
#include <GL/glut.h>
#include <GL/gl.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MASTER_PROC 0
#define GLBL_NUM_ROWS 1000
#define GLBL_NUM_COLS 1000
#define RADIUS_LIMIT 2
#define ORB_COUNT_MAX 50

colour calc_colour(double complex c, double complex grid_point);
double complex map_point(double complex grid_point);
void create_julia_set_image(colour *pixels, int argc, char **argv);

int SaveBitmap(const char *filename, int nX, int nY, int nWidth, int nHeight);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //wait for all processes to start before beginning for timing reasons
    //declare start_time for all procs to avoid compiler warning later
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    double complex c = atof(argv[1]) + atof(argv[2])*I;

    //divide rows among procs
    int num_rows = GLBL_NUM_ROWS / num_procs + ((rank < GLBL_NUM_ROWS % num_procs) ? 1 : 0);

    //allocate array for pixels each proc with process. Master needs to be able to collect all pixels
    colour *pixels = (rank == MASTER_PROC) ? (colour *) malloc(GLBL_NUM_ROWS * GLBL_NUM_COLS * sizeof(colour)) 
        : (colour *) malloc(num_rows * GLBL_NUM_COLS * sizeof(colour));

    //start at top left point in procs range
    double complex grid_point = 0 + (rank * floor(GLBL_NUM_ROWS / num_procs) + fmin(rank, GLBL_NUM_ROWS % num_procs))*I;

    //calculate pixel colour for each point in proc's range
    for (int pixel_row = 0; pixel_row < num_rows; pixel_row++) {
        for (grid_point = 0 + cimag(grid_point)*I; creal(grid_point) < GLBL_NUM_COLS; grid_point += 1) {
            pixels[pixel_row * GLBL_NUM_COLS + (int) creal(grid_point)] = calc_colour(c, grid_point);
        }
        grid_point += 1*I;
    }

    if (rank == MASTER_PROC) {
        int *recv_counts = (int *) malloc(sizeof(int) * num_procs);
        int *displs = (int *) malloc(sizeof(int) * num_procs);

        recv_counts[0] = 3 * num_rows * GLBL_NUM_COLS;
        displs[0] = 0;

        for (int proc = 1; proc < num_procs; proc++) {
            int proc_num_rows = GLBL_NUM_ROWS / num_procs + ((proc < GLBL_NUM_ROWS % num_procs) ? 1 : 0);
         
            recv_counts[proc] = proc_num_rows * GLBL_NUM_COLS * 3;
            displs[proc] = recv_counts[proc - 1] + displs[proc - 1];
        }

        //gather procs' pixel arrays into master
        //will be ordered in recv array (master's pixel array) according to rank, so will be in order
        //MPI_IN_PLACE so that master doesn't send to itself (avoids error of using pixel array as send and recv buffer)
        MPI_Gatherv(MPI_IN_PLACE, recv_counts[0], MPI_FLOAT, pixels, recv_counts, displs, MPI_FLOAT, MASTER_PROC, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(pixels, num_rows * GLBL_NUM_COLS * 3, MPI_FLOAT, NULL, NULL, NULL, NULL, MASTER_PROC, MPI_COMM_WORLD);
    }

    if (rank == MASTER_PROC) {
        double end_time = MPI_Wtime();
        double elapsed_seconds = end_time - start_time;

        printf("\nTook %.3f seconds to calculate the Julia set\n\n", elapsed_seconds);

        create_julia_set_image(pixels, argc, argv);
        //scanf("%d", rank);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

//colour based on exit time when calculating orbit of z0
colour calc_colour(double complex c, double complex grid_point) {
    int orb_point_count = 0;
    double complex z  = map_point(grid_point);
    z = cpow(z, 2) + c;

    while (orb_point_count < ORB_COUNT_MAX && cabs(z) <= RADIUS_LIMIT) {
        orb_point_count++;
        z = cpow(z, 2) + c;
    }

    //0<=orb_point_count<=5 -> colour = 0, ..., 46<=orb_point_count<=50 -> colour = 10
    int colour = (orb_point_count == 0) ? 0 : ceil((float) orb_point_count / (ORB_COUNT_MAX / COLOURS_SIZE)) - 1;

    return COLOURS[colour];
}

//normalize grid_point to be between -2 and 2 (both x and y)
double complex map_point(double complex grid_point) {
    int l = fmin(GLBL_NUM_COLS, GLBL_NUM_ROWS);

    double real = 2 * RADIUS_LIMIT * (creal(grid_point) - GLBL_NUM_COLS / 2.0) / l;
    double imag = 2 * RADIUS_LIMIT * (cimag(grid_point) - GLBL_NUM_ROWS / 2.0) / l;

    return real + imag*I;
}

void SaveFile(){
	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	
	FILE *file = fopen("Julia_Set.bmp", "wb");
	
	unsigned char *image = (unsigned char*)malloc(sizeof(unsigned char)*GLBL_NUM_COLS*GLBL_NUM_ROWS*3);

	if(image!=NULL ){
		if( file!=NULL ){
			glReadPixels( 0, 0, GLBL_NUM_COLS, GLBL_NUM_ROWS, GL_BGR_EXT, GL_UNSIGNED_BYTE, image );

			memset( &bf, 0, sizeof( bf ) );
			memset( &bi, 0, sizeof( bi ) );

			bf.bfType	= 0x4d42;
			bf.bfSize	= sizeof(bf)+sizeof(bi)+GLBL_NUM_COLS*GLBL_NUM_ROWS*3;
			bf.bfOffBits	= sizeof(bf)+sizeof(bi);
			bi.biSize	= sizeof(bi);
			bi.biWidth	= GLBL_NUM_COLS;
			bi.biHeight	= GLBL_NUM_ROWS;
			bi.biPlanes	= 1;
			bi.biBitCount	= 24;
			bi.biSizeImage	= GLBL_NUM_COLS*GLBL_NUM_ROWS*3;

			fwrite( &bf, sizeof(bf), 1, file );
			fwrite( &bi, sizeof(bi), 1, file );
			fwrite( image, sizeof(unsigned char), GLBL_NUM_ROWS*GLBL_NUM_COLS*3, file);

			fclose( file );
		}
		free( image );
	}
}

int SaveBitmap(const char *filename, int nX, int nY, int nWidth, int nHeight) {
	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;

	unsigned char *ptrImage = (unsigned char*) malloc(
			sizeof(unsigned char) * nWidth * nHeight * 3
					+ (4 - (3 * nWidth) % 4) * nHeight);

	if (ptrImage == NULL)
		return EXIT_FAILURE;

	FILE *fptr = fopen(filename, "wb");

	if (fptr == NULL) {
		free(ptrImage);
		return EXIT_FAILURE;
	}

	//read pixels from framebuffer
	glReadPixels(nX, nY, nWidth, nHeight, GL_BGR_EXT, GL_UNSIGNED_BYTE,
			ptrImage);

	// set memory buffer for bitmap header and informaiton header
	memset(&bf, 0, sizeof(bf));
	memset(&bi, 0, sizeof(bi));

	// configure the headers with the give parameters

	bf.bfType = 0x4d42;
	bf.bfSize = sizeof(bf) + sizeof(bi) + nWidth * nHeight * 3
			+ (4 - (3 * nWidth) % 4) * nHeight;
	bf.bfOffBits = sizeof(bf) + sizeof(bi);
	bi.biSize = sizeof(bi);
	bi.biWidth = nWidth + nWidth % 4;
	bi.biHeight = nHeight;
	bi.biPlanes = 1;
	bi.biBitCount = 24;
	bi.biSizeImage = nWidth * nHeight * 3 + (4 - (3 * nWidth) % 4) * nHeight;

	// to files
	fwrite(&bf, sizeof(bf), 1, fptr);
	fwrite(&bi, sizeof(bi), 1, fptr);
	fwrite(ptrImage, sizeof(unsigned char),
			nWidth * nHeight * 3 + (4 - (3 * nWidth) % 4) * nHeight, fptr);
	fclose(fptr);
	free(ptrImage);
	return 1;
}

void create_julia_set_image(colour *pixels, int argc, char **argv) {
    //glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    //glutInitWindowSize(GLBL_NUM_COLS, GLBL_NUM_ROWS);
    //glutCreateWindow("Julia Sets");

    //glEnable (GL_TEXTURE_2D);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    //glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, GLBL_NUM_COLS, GLBL_NUM_ROWS, 0, GL_RGB, GL_FLOAT, &pixels[0]);

    //glBegin(GL_QUADS);
        //glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        //glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        //glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  1.0);
        //glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  1.0);
    //glEnd();

    //SaveBitmap("Julia_Set_Image.bmp", 0, 0, GLBL_NUM_COLS, GLBL_NUM_ROWS);
	
    //glFlush();

    //PPM Stuff
    	FILE * fp;
	fp = fopen("image.ppm", "wb");	
	fprintf(fp, "P6\n%d %d\n255\n", GLBL_NUM_COLS, GLBL_NUM_ROWS);
	
	int iterator = 0;
	
	for(int i = 0; i < GLBL_NUM_ROWS; i++){
		for(int j = 0 ; j < GLBL_NUM_COLS; j++){
			unsigned char colours[3];
			colours[0] = (unsigned char)pixels[iterator].red;
			colours[1] = (unsigned char)pixels[iterator].green;
			colours[2] = (unsigned char)pixels[iterator].blue;
			fwrite(colours, 1, 3, fp);
			iterator++;
		}
	}	
	
	fclose(fp);
    
}
