/*********************************************************************
* Filename:   aes.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding AES implementation.
*********************************************************************/

#ifndef AES_H
#define AES_H

/*************************** HEADER FILES ***************************/
#include <stddef.h>
#include <memory.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <stdbool.h>

/****************************** MACROS ******************************/
#define AES_BLOCK_SIZE 16               // AES operates on 16 bytes at a time

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;            // 8-bit byte
typedef unsigned int WORD;             // 32-bit word, change to "long" for 16-bit machines

#define TRUE  1
#define FALSE 0

/*********************** FUNCTION DECLARATIONS **********************/
///////////////////
// AES
///////////////////
// Key setup must be done before any AES en/de-cryption functions can be used.
void aes_key_setup(const BYTE key[],          // The key, must be 128, 192, or 256 bits
                   WORD w[],                  // Output key schedule to be used later
                   int keysize);              // Bit length of the key, 128, 192, or 256

__device__ void aes_encrypt(const BYTE in[],             // 16 bytes of plaintext
                 BYTE out[],                  // 16 bytes of ciphertext
                 const WORD key[],            // From the key setup
                 int keysize);                // Bit length of the key, 128, 192, or 256

__device__ void aes_decrypt(const BYTE in[],             // 16 bytes of ciphertext
                 BYTE out[],                  // 16 bytes of plaintext
                 const WORD key[],            // From the key setup
                 int keysize);                // Bit length of the key, 128, 192, or 256

///////////////////
// AES - CBC
///////////////////
__global__ void aes_encrypt_cbc(const BYTE in[],          // Plaintext
                    size_t in_len,            // Must be a multiple of AES_BLOCK_SIZE
                    BYTE out[],               // Ciphertext, same length as plaintext
                    const WORD key[],         // From the key setup
                    int keysize,              // Bit length of the key, 128, 192, or 256
                    const BYTE iv[]);         // IV, must be AES_BLOCK_SIZE bytes long

__global__ void aes_decrypt_cbc(const BYTE in[],          // Plaintext
                    size_t in_len,            // Must be a multiple of AES_BLOCK_SIZE
                    BYTE out[],               // Ciphertext, same length as plaintext
                    const WORD key[],         // From the key setup
                    int keysize,              // Bit length of the key, 128, 192, or 256
                    const BYTE iv[]);         // IV, must be AES_BLOCK_SIZE bytes long


///////////////////
// AES - ECB
///////////////////
__global__ void aes_encrypt_ecb(const BYTE in[],             // 16 bytes of plaintext
                     size_t in_len,
                     BYTE out[],                  // 16 bytes of ciphertext
                     const WORD key[],            // From the key setup
                     int keysize);                // Bit length of the key, 128, 192, or 256

__global__ void aes_decrypt_ecb(const BYTE in[],             // 16 bytes of ciphertext
                     size_t in_len,
                     BYTE out[],                  // 16 bytes of plaintext
                     const WORD key[],            // From the key setup
                     int keysize);                // Bit length of the key, 128, 192, or 256


///////////////////
// Loggin Functions
///////////////////
void check_cuda_error(char * msg, cudaError_t err);


///////////////////
// Test functions
///////////////////
int aes_test();
int aes_ecb_test();
int aes_cbc_test();

#endif   // AES_H
