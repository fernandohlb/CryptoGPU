/*********************************************************************
* Filename:   aes_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding AES
              implementation. These tests do not encompass the full
              range of available test vectors and are not sufficient
              for FIPS-140 certification. However, if the tests pass
              it is very, very likely that the code is correct and was
              compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include "aes.h"

/*********************** FUNCTION DEFINITIONS ***********************/
void print_hex(BYTE str[], int len)
{
    int idx;

    for(idx = 0; idx < len; idx++)
        printf("%02x", str[idx]);
}

int aes_ecb_test(char * path_to_file)
{    
    WORD key_schedule[60];
    BYTE * plaintext;
    BYTE * ciphertext;
    BYTE * cipher_bak;
    BYTE * plain_after;
    BYTE key[32] = {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4};

    int pass = 1;

    const char *filename = path_to_file;

    //Key Setup
    aes_key_setup(key, key_schedule, 256);

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

    if (stat(filename, &st) == 0){
        plaintext = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
        ciphertext = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
        cipher_bak = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
        plain_after = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
    };

    FILE *file = fopen(filename, "rb");

    if(!file)
        return 0;

    if(plaintext != NULL && file){
        unsigned long current_byte = 0;

        while(fread(&plaintext[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };

        while(current_byte < plaintext_size - 1){
            current_byte += 1;
            plaintext[current_byte] = 0;
        }
    };

    aes_encrypt_ecb(plaintext, plaintext_size, ciphertext, key_schedule, 256);
    memcpy(cipher_bak, ciphertext, plaintext_size);

    aes_decrypt_ecb(ciphertext, plaintext_size, plain_after, key_schedule, 256);

    FILE *enc_file = fopen("ciphertext_ecb", "wb+");
    FILE *dec_file = fopen("plaintext_ecb", "wb+");

    fwrite(cipher_bak, sizeof(BYTE) * plaintext_size, 1, enc_file);
    fwrite(plain_after, sizeof(BYTE) * plaintext_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);

    return(pass);
}

int aes_cbc_test(char * path_to_file)
{
    WORD key_schedule[60];
    BYTE * plaintext;
    BYTE * ciphertext;
    BYTE * cipher_bak;
    BYTE * plain_after;
    BYTE iv[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
    BYTE key[32] = {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4};
    int pass = 1;

    const char *filename = path_to_file;

    //Key Setup
    aes_key_setup(key, key_schedule, 256);

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

    plaintext = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
    ciphertext = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
    cipher_bak = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);
    plain_after = (BYTE *) malloc(sizeof(BYTE) * plaintext_size);

    FILE *file = fopen(filename, "rb");

    if(!file)
        return 0;

    if(plaintext != NULL && file){
        unsigned long current_byte = 0;

        while(fread(&plaintext[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };

        while(current_byte < plaintext_size - 1){
            current_byte += 1;
            plaintext[current_byte] = 0;
        }
    };

    aes_encrypt_cbc(plaintext, plaintext_size, ciphertext, key_schedule, 256, iv);
    memcpy(cipher_bak, ciphertext, plaintext_size);

    aes_decrypt_cbc(ciphertext, plaintext_size, plain_after, key_schedule, 256, iv);

    FILE *enc_file = fopen("ciphertext_cbc", "wb+");
    FILE *dec_file = fopen("plaintext_cbc", "wb+");

    fwrite(cipher_bak, sizeof(BYTE) * plaintext_size, 1, enc_file);
    fwrite(plain_after, sizeof(BYTE) * plaintext_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);

    return(pass);
}


int aes_test(char * path_to_file)
{
    int pass = 1;

    pass = pass && aes_ecb_test(path_to_file);
    pass = pass && aes_cbc_test(path_to_file);

    return(pass);
}

int main(int argc, char *argv[])
{
    if(argc != 2){
        printf("\nUsage:\n\t %s <relative/path/to/file>\n", argv[0]);
        return 1;
    }

    printf("AES Tests: %s\n", aes_test(argv[1]) ? "SUCCEEDED" : "FAILED");

    return(0);
}
