/*
* Authors: Troy Nechanicky, 150405860; Ben Ngan, 140567260; Alvin Yao, 150580680
* Date: November 16, 2018
*
* Usage: julia_set c_real c_imag
* Description:
*   Displays an image of the Julia set for z^2 + c using points from (0,0) to (1000,1000)
* Compile like: mpicc julia_set.c -o julia_set -std=c99 -lm -lglut -lGL -lGLU
*/

#include <complex.h>
#include "colour.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "GL/glut.h"
#include "GL/gl.h"

#define MASTER_PROC 0
#define GLBL_NUM_ROWS 1000
#define GLBL_NUM_COLS 1000
#define RADIUS_LIMIT 2
#define ORB_COUNT_MAX 50

colour calc_colour(double complex c, double complex grid_point);
double complex map_point(double complex grid_point);
void create_julia_set_image(colour *pixels, int argc, char **argv);

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
    int num_rows = floor(GLBL_NUM_ROWS / num_procs) + fmin(rank, GLBL_NUM_ROWS % num_procs);

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

        for (int i = 1; i < num_procs; i++)
        {
            recv_counts[i] = (i * floor(GLBL_NUM_ROWS / num_procs) + fmin(i, GLBL_NUM_ROWS % num_procs)) * GLBL_NUM_COLS * 3;
            displs[i] = recv_counts[i - 1] + displs[i - 1];
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
        scanf("%d", rank);
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

void create_julia_set_image(colour *pixels, int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(GLBL_NUM_COLS, GLBL_NUM_ROWS);
    glutCreateWindow("Julia Sets");

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0, 1.0, 1.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glEnable(GL_DEPTH_TEST);
    gluOrtho2D(0.0, GLBL_NUM_COLS, GLBL_NUM_ROWS, 0.0);

    int iterator = 0;
    float r, g, b = 0;
    for (int i = 0; i < GLBL_NUM_COLS; i++) {
        for (int j = 0 ; j < GLBL_NUM_ROWS; j++) {
            r = pixels[iterator].red;
            g = pixels[iterator].green;
            b = pixels[iterator].blue;
            glBegin(GL_POINTS);
            glColor3f(r, g, b);
            glVertex2i(i, j);
            glEnd();
            iterator++;
        }
    }

    glFlush();

    //SaveBitmap("output.bmp", 0, 0, GLBL_NUM_COLS, GLBL_NUM_ROWS);
}