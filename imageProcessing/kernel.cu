#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>


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
        if (random % 2) { // rotate 90 degrees
            new_idx = (x * height + (height - y - 1)) * channels;
        }
        else{ // rotate -90 degrees
            new_idx = ((width - x - 1) * height + y) * channels;
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
        /*throw std::runtime_error("Image dimensions must be multiples of 32");*/
        // Resize the image to be multiples of 32
        int new_width = *width - *width % 32;
        int new_height = *height - *height % 32;
        int new_size = new_width * new_height * *channels;
        unsigned char* new_image = new unsigned char[new_size];
        for (int i = 0; i < new_height; i++) {
			for (int j = 0; j < new_width; j++) {
				for (int c = 0; c < *channels; c++) {
					new_image[(i * new_width + j) * *channels + c] = image[(i * *width + j) * *channels + c];
				}
			}
		}
        *width = new_width;
		*height = new_height;
		stbi_image_free(image);
		return new_image;
    }
    return image;
}
#include <windows.h>

void read_directory(const std::string& name, std::vector<std::string>& v)
{
    std::string pattern(name);
    pattern.append("\\*");
    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
        do {
            v.push_back(data.cFileName);
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
}

int main()
{
    srand(time(NULL));
    std::string dir = ".\\imgs";
    std::string dest = ".\\out\\";
    std::vector<std::string> v;
    // read all images in the directory imgs with dir command
    read_directory(dir, v);

    // start time omp parallel
    auto start = omp_get_wtime();

        for (int i = 0; i<v.size(); i++) {
            int width, height, channels;

            std::string img = v[i];
            if (img == "." || img == "..") continue;

            IMG image = readImageFromFile(dir + "\\" + img, &width, &height, &channels);



            IMG gpu_image, out_image;
            assertCudaSuccess(cudaMalloc(&gpu_image, width * height * channels * sizeof(unsigned char)));
            assertCudaSuccess(cudaMalloc(&out_image, width * height * channels * sizeof(unsigned char)));

            assertCudaSuccess(cudaMemcpy(gpu_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 block(BLOCK, BLOCK); // 16x16 block
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); // grid dimensions based on the image size and block size (rounded down)

            blurImage << <grid, block >> > (gpu_image, out_image, width, height, channels);
            assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            stbi_write_jpg((dest + "blur_" + img).c_str(), width, height, channels, image, 100);

            rotateImage << <grid, block >> > (gpu_image, out_image, width, height, channels, rand() * 1000);
            assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            stbi_write_jpg((dest + "rotated_" + img).c_str(), height, width, channels, image, 100);

            negativeImage << <grid, block >> > (gpu_image, out_image, width, height, channels);
            assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            stbi_write_jpg((dest + "negative_" + img).c_str(), width, height, channels, image, 100);

            edgeDetection << <grid, block >> > (gpu_image, out_image, width, height, channels);
            assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            stbi_write_jpg((dest + "edges_" + img).c_str(), width, height, channels, image, 100);

            pixelation << <grid, block >> > (gpu_image, out_image, width, height, channels);
            assertCudaSuccess(cudaMemcpy(image, out_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            stbi_write_jpg((dest + "pixelization_" + img).c_str(), width, height, channels, image, 100);

            //// Free memory
            stbi_image_free(image);
            cudaFree(gpu_image);
            cudaFree(out_image);
    }

    // end time omp parallel
    auto end = omp_get_wtime();
    std::cout << "Time: " << end - start << "s" << std::endl;





    return 0;
}
