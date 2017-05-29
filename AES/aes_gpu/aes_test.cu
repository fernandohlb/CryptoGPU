#include "aes.h"

int aes_ecb_test()
{
    BYTE *h_data;
    BYTE *d_data = NULL;
    BYTE *h_encrypted_data;
    BYTE *h_decrypted_data;
    BYTE *d_encrypted_data;
    BYTE *d_decrypted_data;
    BYTE key[32] = {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4};
    WORD h_key_schedule[60];
    WORD * d_key_schedule;
    const char *filename = "sample_files/hubble_1.tif";

    //Key Setup
    aes_key_setup(key, h_key_schedule, 256);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    struct stat st;

    if (stat(filename, &st) != 0){
        printf("Could not find File");
    };

    unsigned long plaintext_size = 1;
    if(st.st_size % 16 != 0){
        plaintext_size = st.st_size + (16 - st.st_size % 16);
    } else {
        plaintext_size = st.st_size;
    }

    h_data = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);

    FILE *file = fopen(filename, "rb");

    if(h_data != NULL && file){
        unsigned long current_byte = 0;

        while(fread(&h_data[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };

        while(current_byte < plaintext_size - 1){
            current_byte += 1;
            h_data[current_byte] = 0;
        }
    };

    h_encrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    h_decrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    
    //Alocando o BYTE data no device
    err = cudaMalloc((void **)&d_data, sizeof(BYTE) * st.st_size);
    check_cuda_error("Failed to allocate device data", err);    

    //Alocando o BYTE d_encrypted_data no device
    err = cudaMalloc((void **)&d_encrypted_data, sizeof(BYTE) * st.st_size);
    check_cuda_error("Failed to allocate device encrypted_data", err);

    //Alocando o BYTE d_decrypted_data no device
    err = cudaMalloc((void **)&d_decrypted_data, sizeof(BYTE) * st.st_size);
    check_cuda_error("Failed to allocate device decrypted_data", err);

    //Alocando o WORD key no device
    err = cudaMalloc((void **)&d_key_schedule, sizeof(WORD) * 60);
    check_cuda_error("Failed to allocate device data", err);    

    // Copy the host BYTE data in host memory to the device in
    // device memory
    printf("Copy input BYTE data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_data, h_data, sizeof(BYTE) * st.st_size, cudaMemcpyHostToDevice);
    check_cuda_error("Failed to copy BYTE data from host to device", err);

    // Copy the host WORD key_schedule in host memory to the device in
    // device memory
    printf("Copy input BYTE data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_key_schedule, h_key_schedule, sizeof(WORD) * 60, cudaMemcpyHostToDevice);
    check_cuda_error("Failed to copy BYTE data from host to device", err);

    // Launch the aes_encrypt CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =((sizeof(BYTE) * st.st_size)/AES_BLOCK_SIZE/threadsPerBlock)+1;
   
    //Launch encryption
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    aes_encrypt_ecb<<<blocksPerGrid, threadsPerBlock>>>(d_data, st.st_size, d_encrypted_data, d_key_schedule, 256);
    // my_test<<<blocksPerGrid, threadsPerBlock>>>();
    err = cudaGetLastError();
    check_cuda_error("Failed to launch aes_encrypt kernel", err);

    // Copy the device result encrypted_data in device memory to the host result vector
    // in host memory.
    printf("Copy output encrypted_data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_encrypted_data, d_encrypted_data, sizeof(BYTE) * st.st_size, cudaMemcpyDeviceToHost);
    check_cuda_error("Failed to copy vector encrypted_data from device to host", err);


    //Launch decryption
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    aes_decrypt_ecb<<<blocksPerGrid, threadsPerBlock>>>(d_encrypted_data, st.st_size, d_decrypted_data, d_key_schedule, 256);
    err = cudaGetLastError();
    check_cuda_error("Failed to launch aes_decrypt kernel", err);

    // Copy the device result decrypted_data in device memory to the host result vector
    // in host memory.
    printf("Copy output decrypted_data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_decrypted_data, d_decrypted_data, sizeof(BYTE) * st.st_size, cudaMemcpyDeviceToHost);
    check_cuda_error("Failed to copy vector decrypted_data from device to host", err);

    FILE *enc_file = fopen("ciphertext_ecb", "wb+");
    FILE *dec_file = fopen("plaintext_ecb", "wb+");

    fwrite(h_encrypted_data, sizeof(BYTE) * st.st_size, 1, enc_file);
    fwrite(h_decrypted_data, sizeof(BYTE) * st.st_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);

    return TRUE;
};

int aes_cbc_test()
{
    BYTE *h_data;
    BYTE *d_data = NULL;
    BYTE *h_encrypted_data;
    BYTE *h_decrypted_data;
    BYTE *d_encrypted_data;
    BYTE *d_decrypted_data;
    BYTE *d_iv;
    BYTE h_iv[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
    BYTE key[32] = {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4};
    WORD h_key_schedule[60];
    WORD * d_key_schedule;
    const char *filename = "sample_files/hubble_1.tif";

    //Key Setup
    aes_key_setup(key, h_key_schedule, 256);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    struct stat st;

    if (stat(filename, &st) != 0){
        printf("Could not find File");
    };

    unsigned long plaintext_size = 1;
    if(st.st_size % 16 != 0){
        plaintext_size = st.st_size + (16 - st.st_size % 16);
    } else {
        plaintext_size = st.st_size;
    }

    h_data = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);

    FILE *file = fopen(filename, "rb");

    if(h_data != NULL && file){
        unsigned long current_byte = 0;

        while(fread(&h_data[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };

        while(current_byte < plaintext_size - 1){
            current_byte += 1;
            h_data[current_byte] = 0;
        }
    };

    h_encrypted_data = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
    h_decrypted_data = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
    
    //Alocando o BYTE data no device
    err = cudaMalloc((void **)&d_data, sizeof(BYTE) * plaintext_size);
    check_cuda_error("Failed to allocate device data", err);    

    //Alocando o BYTE d_encrypted_data no device
    err = cudaMalloc((void **)&d_encrypted_data, sizeof(BYTE) * plaintext_size);
    check_cuda_error("Failed to allocate device encrypted_data", err);

    //Alocando o BYTE d_decrypted_data no device
    err = cudaMalloc((void **)&d_decrypted_data, sizeof(BYTE) * plaintext_size);
    check_cuda_error("Failed to allocate device decrypted_data", err);

    //Alocando o WORD key no device
    err = cudaMalloc((void **)&d_key_schedule, sizeof(WORD) * 60);
    check_cuda_error("Failed to allocate device data", err);

    //Alocando o BYTE IV no device
    err = cudaMalloc((void **)&d_iv, sizeof(BYTE) * AES_BLOCK_SIZE);
    check_cuda_error("Failed to allocate device data", err);

    // Copy the host BYTE data in host memory to the device in
    // device memory
    printf("Copy input BYTE data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_data, h_data, sizeof(BYTE) * plaintext_size, cudaMemcpyHostToDevice);
    check_cuda_error("Failed to copy BYTE data from host to device", err);

    // Copy the host WORD key_schedule in host memory to the device in
    // device memory
    printf("Copy input BYTE data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_key_schedule, h_key_schedule, sizeof(WORD) * 60, cudaMemcpyHostToDevice);
    check_cuda_error("Failed to copy BYTE data from host to device", err);

    // Copy the host WORD key_schedule in host memory to the device in
    // device memory
    printf("Copy input BYTE data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_iv, h_iv, sizeof(BYTE) * AES_BLOCK_SIZE, cudaMemcpyHostToDevice);
    check_cuda_error("Failed to copy BYTE data from host to device", err);

    // Launch the aes_encrypt CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =((sizeof(BYTE) * plaintext_size)/AES_BLOCK_SIZE/threadsPerBlock)+1;
   
    //Launch encryption
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    aes_encrypt_cbc<<<blocksPerGrid, threadsPerBlock>>>(d_data, plaintext_size, d_encrypted_data, d_key_schedule, 256, d_iv);
    // my_test<<<blocksPerGrid, threadsPerBlock>>>();
    err = cudaGetLastError();
    check_cuda_error("Failed to launch aes_encrypt kernel", err);

    // Copy the device result encrypted_data in device memory to the host result vector
    // in host memory.
    printf("Copy output encrypted_data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_encrypted_data, d_encrypted_data, sizeof(BYTE) * plaintext_size, cudaMemcpyDeviceToHost);
    check_cuda_error("Failed to copy vector encrypted_data from device to host", err);


    //Launch decryption
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    aes_decrypt_cbc<<<blocksPerGrid, threadsPerBlock>>>(d_encrypted_data, plaintext_size, d_decrypted_data, d_key_schedule, 256, d_iv);
    err = cudaGetLastError();
    check_cuda_error("Failed to launch aes_decrypt kernel", err);

    // Copy the device result decrypted_data in device memory to the host result vector
    // in host memory.
    printf("Copy output decrypted_data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_decrypted_data, d_decrypted_data, sizeof(BYTE) * plaintext_size, cudaMemcpyDeviceToHost);
    check_cuda_error("Failed to copy vector decrypted_data from device to host", err);

    FILE *enc_file = fopen("ciphertext_cbc", "wb+");
    FILE *dec_file = fopen("plaintext_cbc", "wb+");

    fwrite(h_encrypted_data, sizeof(BYTE) * plaintext_size, 1, enc_file);
    fwrite(h_decrypted_data, sizeof(BYTE) * plaintext_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);

    return TRUE;
};


int aes_test()
{
    int pass = 1;

    // pass = pass && aes_ecb_test();
    pass = pass && aes_cbc_test();

    return(pass);
}

int main(int argc, char *argv[])
{
    printf("AES Tests: %s\n", aes_test() ? "SUCCEEDED" : "FAILED");

    return(0);
}