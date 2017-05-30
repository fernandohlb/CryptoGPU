/*********************************************************************
* Filename:   rot-13_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ROT-13
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "../../include/time_utils.h"


/*********************** FUNCTION DEFINITIONS ***********************/

__global__ void rot13(char str_in[], long size)
{
   int case_type;
   int idx = blockDim.x * blockIdx.x + threadIdx.x;

   if (idx < size) {
      // Only process alphabetic characters.
      if (str_in[idx] < 'A' || (str_in[idx] > 'Z' && str_in[idx] < 'a') || str_in[idx] > 'z')
         return;
      // Determine if the char is upper or lower case.
      if (str_in[idx] >= 'a')
         case_type = 'a';
      else
         case_type = 'A';
      // Rotate the char's value, ensuring it doesn't accidentally "fall off" the end.
      str_in[idx] = (str_in[idx] + 13) % (case_type + 26);
      if (str_in[idx] < 26)
         str_in[idx] += case_type;
   }

}





int rot13_test(char filename[],int numThreads)
{
    struct timespec wc_start[STEPS_SIZE], wc_end[STEPS_SIZE];
    double cpu_start[STEPS_SIZE], cpu_end[STEPS_SIZE];
    start_timers(cpu_start, wc_start, alloc);
    char *data = NULL;
    char * data_buf = NULL;
    char *encrypted_data;
    char *decrypted_data;
    int pass = 1;
    //const char *filename = "sample_files/king_james_bible.txt";
    struct stat st;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    if (stat(filename, &st) == 0){
        data = (char *) malloc(sizeof(char) * st.st_size);
    };

    FILE *file = fopen(filename, "rb");

    if(data != NULL && file){
        int current_byte = 0;

        while(fread(&data[current_byte], sizeof(char), 1, file) == 1){
            current_byte += 1;
        };
    };

    int size = sizeof(char) * st.st_size;
    encrypted_data = (char *) malloc(size);
    decrypted_data = (char *) malloc(size);
    data_buf = (char *) malloc(size);


    // To encode, just apply ROT-13.
    //strcpy(data_buf, data);

    // Allocate the device input vector A
    char *d_data_buf = NULL;
    err = cudaMalloc((void **)&d_data_buf, size);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    // printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_data_buf, data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    end_timers(cpu_end, wc_end, alloc);


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = numThreads;
    int blocksPerGrid = (size/threadsPerBlock)+1;
    printf("Tamanho do Arquivo: %i \n",size);
    start_timers(cpu_start, wc_start, calc);
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    rot13<<<blocksPerGrid, threadsPerBlock>>>(d_data_buf,size);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(data_buf, d_data_buf, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //rot13(data_buf);
 
    strcpy(encrypted_data, data_buf);

    // Launch the Vector Add CUDA Kernel
    //int threadsPerBlock = 256;
    //int blocksPerGrid =(sizeof(char) * st.st_size + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    rot13<<<blocksPerGrid, threadsPerBlock>>>(d_data_buf,size);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    end_timers(cpu_end, wc_end, calc);


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(data_buf, d_data_buf, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //rot13(data_buf);
    strcpy(decrypted_data, data_buf);


    //Compara o dado Decriptografado com o Dado Original
    pass = pass && !strcmp(decrypted_data, data);

    start_timers(cpu_start, wc_start, ioops);
    FILE *enc_file = fopen("text_enc.txt", "wb+");
    FILE *dec_file = fopen("text_dec.txt", "wb+");

    fwrite(encrypted_data, size, 1, enc_file);
    fwrite(decrypted_data, size, 1, dec_file);
    end_timers(cpu_end, wc_end, ioops);

    print_elapsed(cpu_start, wc_start, cpu_end, wc_end);


    // Free device global memory
    err = cudaFree(d_data_buf);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device string d_str_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return(pass);
}

int main(int argc, char *argv[ ])
{
    int numThreads = strtol(argv[2], NULL, 10);
    printf("ROT-13 GPU tests\n");
    rot13_test(argv[1],numThreads);
    // printf("ROT-13 GPU tests: %s\n", rot13_test(argv[1],numThreads) ? "SUCCEEDED" : "FAILED");

    return(0);
}
