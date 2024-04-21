#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 



using IMG = decltype(stbi_load(std::declval<const char*>(), std::declval<int*>(), std::declval<int*>(), std::declval<int*>(), std::declval<int>()));
constexpr int BLOCK = 16;


__global__ void blurImage(const IMG image, IMG out_image,  int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfWindowSize = 10;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        int sum_r = 0;
        int sum_g = 0;
        int sum_b = 0;
        for (int i = -halfWindowSize; i <= halfWindowSize; i++) { // iterate over the 5x5 kernel
            for (int j = -halfWindowSize; j <= halfWindowSize; j++) { // iterate over the 5x5 kernel
                int x1 = x + i; // x coordinate of the kernel
                int y1 = y + j; // y coordinate of the kernel
                if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) { // check if the kernel is within the image
                    int idx1 = (y1 * width + x1) * channels; // index of the kernel pixel

                    // sum the colors
                    sum_r += image[idx1];
                    sum_g += image[idx1 + 1];
                    sum_b += image[idx1 + 2];
                }
            }
        }

        // average the colors
        out_image[idx] = sum_r / (halfWindowSize * 2 + 1) / (halfWindowSize * 2 + 1);
        out_image[idx + 1] = sum_g / (halfWindowSize * 2 + 1) / (halfWindowSize * 2 + 1);
        out_image[idx + 2] = sum_b / (halfWindowSize * 2 + 1) / (halfWindowSize * 2 + 1);
    }
}

// rotate image
__global__ void rotateImage(const IMG image, IMG out_image, int width, int height, int channels, int random) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        int new_idx = 0;
        if (random % 3 == 0) {
            new_idx = (x * height + (height - y - 1)) * channels;
        }
        else if (random % 3 == 1) {
            new_idx = ((width - x - 1) * height + y) * channels;
        }
        else if (random % 3 == 2) {
            new_idx = ((height - y - 1) * width + (width - x - 1)) * channels;
        }
        else {
            new_idx = (y * width + x) * channels;
        }
        out_image[new_idx] = image[idx];
        out_image[new_idx + 1] = image[idx + 1];
        out_image[new_idx + 2] = image[idx + 2];
    }
}

__global__ void negativeImage(const IMG image, IMG out_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        out_image[idx] = 255 - image[idx];
        out_image[idx + 1] = 255 - image[idx + 1];
        out_image[idx + 2] = 255 - image[idx + 2];
    }
}

__global__ void edgeDetection(const IMG image, unsigned char* out_image, int width, int height, int channels) {
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int filter[3][3] = { // Sobel filter for edge detection
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    if (x >= width || y >= height) return;

    // Compute the linear index for the current pixel
    int index = (y * width + x) * channels;

    // Apply filter to each channel
    for (int c = 0; c < channels; c++) {
        int sum = 0;
        for (int fx = -1; fx <= 1; fx++) { // Iterate over the filter window in x direction (columns)
            for (int fy = -1; fy <= 1; fy++) { // Iterate over the filter window in y direction (rows)
                int ix = x + fx; // Compute neighbor index in x direction (columns)
                int iy = y + fy; // Compute neighbor index in y direction (rows)
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) { // Check for boundary conditions
                    int neighborIndex = (iy * width + ix) * channels; // Compute the linear index of the neighbor pixel
                    sum += image[neighborIndex + c] * filter[fx + 1][fy + 1]; // Apply the filter to the neighbor pixel and accumulate the result to the sum
                }
            }
        }
        // Assign the computed value to output image
        // Clamp the result to be between 0 and 255
        out_image[index + c] = sum < 0 ? 0 : (sum > 255 ? 255 : sum);
    }
}

__global__ void pixelation(const IMG input_image, unsigned char* output_image, int width, int height, int channels) {
    int pixelation_size = 8;

    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_index >= width || y_index >= height) return;

    // Calculate the starting pixel of the current pixelation block
    int x_block_start = (x_index / pixelation_size) * pixelation_size; 
    int y_block_start = (y_index / pixelation_size) * pixelation_size;

    // Calculate the average color of the block
    int red_total = 0, green_total = 0, blue_total = 0;
    int block_area = pixelation_size * pixelation_size;
    for (int i = 0; i < pixelation_size; ++i) {
        for (int j = 0; j < pixelation_size; ++j) {
            int current_x = x_block_start + i;
            int current_y = y_block_start + j;
            if (current_x < width && current_y < height) {
                int idx = (current_y * width + current_x) * channels; 
                red_total += input_image[idx];
                green_total += input_image[idx + 1];
                blue_total += input_image[idx + 2];
            }
        }
    }
    int red_avg = red_total / block_area;
    int green_avg = green_total / block_area;
    int blue_avg = blue_total / block_area;

    // Assign the average color to each pixel in the block
    for (int i = 0; i < pixelation_size; ++i) {
        for (int j = 0; j < pixelation_size; ++j) {
            int current_x = x_block_start + i;
            int current_y = y_block_start + j;
            if (current_x < width && current_y < height) {
                int idx = (current_y * width + current_x) * channels;
                output_image[idx] = red_avg;
                output_image[idx + 1] = green_avg;
                output_image[idx + 2] = blue_avg;
            }
        }
    }
}


void assertCudaSuccess(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
        exit(code);
    }
}

IMG readImageFromFile(const std::string& filename, int *width, int *height, int *channels)
{
    IMG image = stbi_load(filename.c_str(), width, height, channels, 0);
    if (!image) {
        throw std::runtime_error(stbi_failure_reason());
    }
    if (*width % 32 != 0 || *height % 32 != 0) {
        throw std::runtime_error("Image dimensions must be multiples of 32");
    }
    return image;
}


int main()
{
    srand(time(NULL));
    int width, height, channels;
    IMG image = readImageFromFile("img.jpg", &width, &height, &channels);

    IMG gpu_image, out_image;
    assertCudaSuccess(cudaMalloc(&gpu_image, width * height * channels * sizeof(unsigned char)));
    assertCudaSuccess(cudaMalloc(&out_image, width * height * channels * sizeof(unsigned char)));

    assertCudaSuccess(cudaMemcpy(gpu_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 block(BLOCK, BLOCK); // 16x16 block
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); // grid dimensions based on the image size and block size (rounded down)

    blurImage << <grid, block >> > (gpu_image, out_image, width, height, channels);
    assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_jpg("img_blur.jpg", width, height, channels, image, 100);

    rotateImage << <grid, block >> > (gpu_image, out_image, width, height, channels, rand()*1000);
    assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_jpg("img_rotated.jpg", height, width, channels, image, 100);

    negativeImage << <grid, block >> > (gpu_image, out_image, width, height, channels);
    assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_jpg("img_negative.jpg", width, height, channels, image, 100);

    edgeDetection << <grid, block >> > (gpu_image, out_image, width, height, channels);
    assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_jpg("img_edges.jpg", width, height, channels, image, 100);

    pixelation << <grid, block >> > (gpu_image, out_image, width, height, channels);
    assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_jpg("img_pixelization.jpg", width, height, channels, image, 100);

    //// Free memory
    stbi_image_free(image);
    cudaFree(gpu_image);




    return 0;
}
