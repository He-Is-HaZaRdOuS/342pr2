/**
 *
 * CENG342 Project-2
 *
 * Edge Detection
 *
 * Usage: executable <input.jpg> <output.jpg> <threadCount> <sequential_output.jpg>
 *
 * @group_id 07
 * @author  Yousif
 * @author  Türker
 * @author  Eren
 * @author  Aysara
 *
 * @version 1.0, 18 May 2024
 */

// ReSharper disable CppUseAuto
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "mpi.h"
#include "omp.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define KERNEL_DIMENSION 3
#define THRESHOLD 40
#define USE_THRESHOLD 0

//Do not use global variables

/* Function prototypes */
void seq_edgeDetection(uint8_t *input_image, int width, int height, int threadCount);
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION], float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]);

int main(int argc,char* argv[]) {
    /* Abort if # of CLA is invalid */
    if(argc != 5){
        std::cerr << "Invalid number of arguments, aborting...\n";
        std::cerr << "Try ./omp <input.jpg> <output.jpg> <threadCount> <sequential.jpg>\n";
        exit(1);
    }

    /* Init MPI just to use its timer functionality */
    MPI_Init(nullptr, nullptr);
    int width, height, bpp;

    /* Prepend path to input and output filenames */
    std::string inputPath = RESOURCES_PATH;
    std::string outputPath = OMP_OUTPUT_PATH;
    inputPath = inputPath + argv[1];
    outputPath = outputPath + argv[2];
    int threadCount = std::stoi(argv[3]);

    /* Read image in grayscale */
    uint8_t *input_image = stbi_load(inputPath.c_str(), &width, &height, &bpp, CHANNEL_NUM);

    if(stbi_failure_reason()) {
        std::cerr << stbi_failure_reason() << " \"" + inputPath + "\"\n";
        std::cerr << "Aborting...\n";
        exit(1);
    }

    printf("Width: %d  Height: %d  BPP: %d \n",width, height, bpp);
    printf("Input: %s , Output: %s  \n",inputPath.c_str(), outputPath.c_str());

    /* Start the timer */
    const double time1= MPI_Wtime();

    seq_edgeDetection(input_image, width, height, threadCount);

    /* Stop the timer */
    const double time2= MPI_Wtime();
    printf("Elapsed time: %lf \n",time2-time1);

    stbi_write_jpg(outputPath.c_str(), width, height, CHANNEL_NUM, input_image, 100);
    stbi_image_free(input_image);

    MPI_Finalize();

    /* Check if the two image outputs are identical */
    {
        /* Prepend path to input and output filenames */
        std::string alt_input = SEQUENTIAL_OUTPUT_PATH;
        std::string par_input = OMP_OUTPUT_PATH;
        alt_input = alt_input + argv[4];
        par_input = par_input + argv[2];
        uint8_t *alt_img, *par_img;
        int seq_width, seq_height, seq_bpp;
        int par_width, par_height, par_bpp;

        /* Read image in grayscale */
        alt_img = stbi_load(alt_input.c_str(), &seq_width, &seq_height, &seq_bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + alt_input + "\"\n";
            std::cerr << "Aborting...\n";
            exit(1);
        }

        par_img = stbi_load(par_input.c_str(), &par_width, &par_height, &par_bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + par_input + "\"\n";
            std::cerr << "Aborting...\n";
            stbi_image_free(alt_img);
            exit(1);
        }

        std::cout << "Comparing " << alt_input << " and " << par_input << std::endl;

        /* Make sure Local and Alternate outputs are the same */
        int err_cnt = 0;
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                if(par_img[x + y * width] != alt_img[x + y * width]) {
                    ++err_cnt;
                }
            }
        }
        if(err_cnt == 0)
            std::cout << "OMP and Sequential images are identical\n";
        else
            std::cout << err_cnt << " pixels are mismatched\n";

        /* Let go of STB image buffers */
        stbi_image_free(alt_img);
        stbi_image_free(par_img);
    }

    return 0;
}

/* Apply Sobel's Operator  */
void seq_edgeDetection(uint8_t *input_image, const int width, const int height, const int threadCount) {
    /* Declare Kernels */
    float sobelX[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {-1, 0, 1},{-2, 0, 2},{-1, 0, 1} };
    float sobelY[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {-1, -2, -1},{0, 0, 0},{1, 2, 1} };
    float slider[KERNEL_DIMENSION][KERNEL_DIMENSION];

    /* Allocate temporary memory to construct final image */
    uint8_t *output_image = static_cast<uint8_t *>(malloc(width * height * sizeof(uint8_t))); // NOLINT(*-use-auto)

    /* Declare vars ahead of time to privatize later */
    int xIndex, yIndex, x, y, wx, wy;

    /* Iterate through all pixels */
    #pragma omp parallel for collapse(2) private(x, wy, wx, xIndex, yIndex, slider) num_threads(threadCount)
    for(y = 0; y < height; ++y) {
        for(x = 0; x < width; ++x) {
            for(wy = 0; wy < KERNEL_DIMENSION; ++wy) {
                for(wx = 0; wx < KERNEL_DIMENSION; ++wx) {
                    xIndex = (x + wx - 1);
                    yIndex = (y + wy - 1);

                    /* Clamp */
                    if(xIndex < 0)
                        xIndex = -xIndex;
                    if(yIndex < 0)
                        yIndex = -yIndex;
                    if(xIndex >= width)
                        xIndex = width - 1;
                    if(yIndex >= height)
                        yIndex = height - 1;

                    /* Build up sliding window */
                    slider[wy][wx] = input_image[xIndex + yIndex * width];
                }
            }

            /* Convolve sliding window with kernels (Sobel X and Y gradient) */
            const float gx = convolve(slider, sobelX);
            const float gy = convolve(slider, sobelY);
            float magnitude = sqrtf(gx * gx + gy * gy);

#if USE_THRESHOLD
            /* Clamp down color values if below THRESHOLD */
            output_image[x + y * width] = magnitude > THRESHOLD ? 255 : 0;
#else
            /* Otherwise use whatever value outputted from square root */
            output_image[x + y * width] = static_cast<uint8_t>(magnitude);
#endif
        }
    }

    /* memcpy final image data to input buffer */
    memcpy(input_image, output_image, width * height * sizeof(uint8_t ));

    /* De-allocate temporary memory */
    free(output_image);
}

/* Convolve slider across kernel (multiply and sum values of two arrays) */
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION], float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]) {
    float sum = 0;
    for(int y = 0; y < KERNEL_DIMENSION; ++y) {
        for(int x = 0; x < KERNEL_DIMENSION; ++x) {
            sum = sum + slider[x][y] * kernel[x][y];
        }
    }

    return sum;
}
