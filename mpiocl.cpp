#include <iostream>
#include <CL/cl.h>
#include <ctime>
#include <cstdlib>

#define ARRAY_SIZE 100

using namespace std;

void quicksort(int *array, int left, int right) {
    int i = left, j = right;
    int pivot = array[(left + right) / 2];

    while (i <= j) {
        while (array[i] < pivot)
            i++;
        while (array[j] > pivot)
            j--;
        if (i <= j) {
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
            i++;
            j--;
        }
    }

    if (left < j)
        quicksort(array, left, j);
    if (i < right)
        quicksort(array, i, right);
}

int main() {
    clock_t start_time = clock();

    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    const char *source = "#define CL_TARGET_OPENCL_VERSION 300 \n\
                            kernel void quicksort(global int *array, int left, int right) \
                            { \
                                int i = left, j = right; \
                                int pivot = array[(left + right) / 2]; \
                                \
                                while (i <= j) { \
                                    while (array[i] < pivot) i++; \
                                    while (array[j] > pivot) j--; \
                                    if (i <= j) { \
                                        int temp = array[i]; \
                                        array[i] = array[j]; \
                                        array[j] = temp; \
                                        i++; \
                                        j--; \
                                    } \
                                } \
                                \
                                if (left < j) quicksort(array, left, j); \
                                if (i < right) quicksort(array, i, right); \
                            }";

    program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "quicksort", &err);

    // Generate random array
    int array[ARRAY_SIZE];
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }

    // Create OpenCL buffer for the input array and write data to it
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, &err);
    err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, array, 0, NULL, NULL);

    // Set kernel arguments and launch kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);
    err = clSetKernelArg(kernel, 1, sizeof(int), (void *)0);
    err = clSetKernelArg(kernel, 2, sizeof(int), (void *)(ARRAY_SIZE - 1));

    size_t globalWorkSize[1] = {ARRAY_SIZE};
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // Read the result from the device
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, array, 0, NULL, NULL);

    clock_t end_time = clock();

    double duration = (double)(end_time - start_time) / ((double)CLOCKS_PER_SEC / 1000000);

    // Print the sorted array
    cout << "Sorted array: ";
    for (int i = 0; i < ARRAY_SIZE; i++) {
        cout << array[i] << " ";
    }
    cout << endl;

    cout << "Time taken: " << duration << " microseconds" << endl;

    // Release OpenCL resources
    err = clFlush(queue);
    err = clFinish(queue);
    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(buffer);
    err = clReleaseCommandQueue(queue);
    err = clReleaseContext(context);

    return 0;
}
