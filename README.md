## Projects

### **Image Processing**

**Description:**
This project generates training datasets for neural networks that recognize images. The program takes an original image as input and produces various modifications, such as blurring, edge detection, color inversion, and reflections along axes using CUDA.

![vermeer005](https://github.com/user-attachments/assets/f9421c35-9802-47b8-9a47-68a3b91d44d0)

#### **Effects**

1. **Blurring:**
    - The original image is scanned with a 5x5 window, calculating the average RGB values for each pixel and applying them to the output image.

![blur_vermeer005](https://github.com/user-attachments/assets/c5eb499f-691d-4118-8f00-a7bd661b0584)

2. **Rotation:**
    - Pixels are repositioned based on random rotations of ±90°.

![rotated_vermeer005](https://github.com/user-attachments/assets/741c765f-0468-4600-b64a-cec26a471d94)

3. **Negative:**
    - Each pixel's value is calculated as dest = 255 - src

![negative_vermeer005](https://github.com/user-attachments/assets/11b8779a-bd09-4546-9504-a3ddfbb4afa3)

4. **Edge Detection:**
    - A simplified Sobel filter is applied:

![edges_vermeer005](https://github.com/user-attachments/assets/9c9cb047-833c-4316-aa46-465ce121153d)

```cpp
int filter[3][3] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};
```

Pixel values are clamped between[0, 255].

5. **Pixelation:**
    - Similar to blurring but applied to larger blocks defined by the `pixelation_size` parameter.

![pixelization_vermeer005](https://github.com/user-attachments/assets/7cd9098b-d430-40e8-9559-b004f3f6360e)

#### **Performance Comparison**

- Execution time depends on the window size used during CUDA kernel execution.
- Key observations:
    - Initial drop in execution time as window size increases.
    - Gradual optimization with larger windows.
    - Stabilization after reaching optimal window size (16).

Example CUDA kernel:

```cpp
dim3 block(BLOCK, BLOCK);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
blurImage <<<grid, block>>> (gpu_image, out_image, width, height, channels);
```
