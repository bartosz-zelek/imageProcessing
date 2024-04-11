#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Kernel function to convert image to grayscale
__global__ void imageToGrayscale(unsigned char* image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = image[idx];
        unsigned char g = image[idx + 1];
        unsigned char b = image[idx + 2];

        unsigned char gray = 0.21f * r + 0.71f * g + 0.07f * b;

        image[idx] = gray;
        image[idx + 1] = gray;
        image[idx + 2] = gray;
    }
}


int main()
{
    int width, height, channels;
    auto image = stbi_load("img.jpg", &width, &height, &channels, 0);
    if (!image) {
        std::cout << "Error: image reading" << std::endl;
        return 1;
    }

    unsigned char* gpu_image;
    cudaMalloc(&gpu_image, width * height * channels * sizeof(unsigned char));

    cudaMemcpy(gpu_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    imageToGrayscale << <grid, block >> > (gpu_image, width, height, channels);

    cudaMemcpy(image, gpu_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("img_gray.jpg", width, height, channels, image, 100);


    return 0;
}
