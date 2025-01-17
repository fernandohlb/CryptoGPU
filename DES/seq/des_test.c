/*********************************************************************
* Filename:   des_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding DES
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
#include "des.h"
#include "../../include/time_utils.h"


char * nomeArquivo;

/*********************** FUNCTION DEFINITIONS ***********************/
int des_test()
{
    BYTE pt1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xE7};
    BYTE pt2[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE pt3[DES_BLOCK_SIZE] = {0x54,0x68,0x65,0x20,0x71,0x75,0x66,0x63};
    BYTE ct1[DES_BLOCK_SIZE] = {0xc9,0x57,0x44,0x25,0x6a,0x5e,0xd3,0x1d};
    BYTE ct2[DES_BLOCK_SIZE] = {0x85,0xe8,0x13,0x54,0x0f,0x0a,0xb4,0x05};
    BYTE ct3[DES_BLOCK_SIZE] = {0xc9,0x57,0x44,0x25,0x6a,0x5e,0xd3,0x1d};
    BYTE ct4[DES_BLOCK_SIZE] = {0xA8,0x26,0xFD,0x8C,0xE5,0x3B,0x85,0x5F};
    BYTE key1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE key2[DES_BLOCK_SIZE] = {0x13,0x34,0x57,0x79,0x9B,0xBC,0xDF,0xF1};
    BYTE three_key1[DES_BLOCK_SIZE * 3] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                           0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                           0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE three_key2[DES_BLOCK_SIZE * 3] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                           0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,0x01,
                                           0x45,0x67,0x89,0xAB,0xCD,0xEF,0x01,0x23};

    BYTE schedule[16][6];
    BYTE three_schedule[3][16][6];
    BYTE buf[DES_BLOCK_SIZE];
    int pass = 1;

    des_key_setup(key1, schedule, true);
    des_crypt(pt1, buf, schedule);
    pass = pass && !memcmp(ct1, buf, DES_BLOCK_SIZE);

    des_key_setup(key1, schedule, false);
    des_crypt(ct1, buf, schedule);
    pass = pass && !memcmp(pt1, buf, DES_BLOCK_SIZE);

    des_key_setup(key2, schedule, true);
    des_crypt(pt2, buf, schedule);
    pass = pass && !memcmp(ct2, buf, DES_BLOCK_SIZE);

    des_key_setup(key2, schedule, false);
    des_crypt(ct2, buf, schedule);
    pass = pass && !memcmp(pt2, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key1, three_schedule, true);
    three_des_crypt(pt1, buf, three_schedule);
    pass = pass && !memcmp(ct3, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key1, three_schedule, false);
    three_des_crypt(ct3, buf, three_schedule);
    pass = pass && !memcmp(pt1, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key2, three_schedule, true);
    three_des_crypt(pt3, buf, three_schedule);
    pass = pass && !memcmp(ct4, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key2, three_schedule, false);
    three_des_crypt(ct4, buf, three_schedule);
    pass = pass && !memcmp(pt3, buf, DES_BLOCK_SIZE);

    return(pass);
}

void enc_dec_file(char filename[])
{
    struct timespec wc_start[STEPS_SIZE], wc_end[STEPS_SIZE];
    double cpu_start[STEPS_SIZE], cpu_end[STEPS_SIZE];
    start_timers(cpu_start, wc_start, alloc);
    BYTE *data;
    BYTE *encrypted_data;
    BYTE *decrypted_data;
    /*char filename[] = "sample_files/";
    	strcat(filename,nomeArquivo);
    	strcat(filename,".tif");*/
    
    //const char filename[] = "sample_files/hubble_1.tif";

    

    BYTE key1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE schedule[16][6];

    struct stat st;

    if (stat(filename, &st) == 0){
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    };

    FILE *file = fopen(filename, "rb");

    if(data != NULL && file){
        int current_byte = 0;

        while(fread(&data[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };
    };

    encrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    decrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);

    BYTE data_buf[DES_BLOCK_SIZE];
    BYTE data_enc[DES_BLOCK_SIZE];
    BYTE data_dec[DES_BLOCK_SIZE];
    printf("Tamanho do Arquivo: %i \n",(int)st.st_size);
    end_timers(cpu_end, wc_end, alloc);

    start_timers(cpu_start, wc_start, calc);
    for(int i = 0; i < st.st_size; i++){
        //printf("Valor de i antes: %i \n", i);
        for(int j = 0; j < DES_BLOCK_SIZE; j++){
            if(i < st.st_size){
                data_buf[j] = data[i];
                i++;
            };
        };

        des_key_setup(key1, schedule, true);
        des_crypt(data_buf, data_enc, schedule);

        des_key_setup(key1, schedule, false);
        des_crypt(data_enc, data_dec, schedule);

        i -= DES_BLOCK_SIZE;
        for(int k = 0; k < DES_BLOCK_SIZE; k++){
            if(i < st.st_size){
                encrypted_data[i] = data_enc[k];
                decrypted_data[i] = data_dec[k];
                i++;
            };
        };
	//i = i+8;
        
        /*
        printf("Valor de i: %i \n", i);*/
        i--;
    };
    end_timers(cpu_end, wc_end, calc);

    start_timers(cpu_start, wc_start, ioops);
    FILE *enc_file = fopen("hubble_enc.tif", "wb+");
    FILE *dec_file = fopen("hubble_dec.tif", "wb+");


    fwrite(encrypted_data, sizeof(BYTE) * st.st_size, 1, enc_file);
    fwrite(decrypted_data, sizeof(BYTE) * st.st_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);
    end_timers(cpu_end, wc_end, ioops);

    print_elapsed(cpu_start, wc_start, cpu_end, wc_end);
};

int main(int argc, char *argv[ ])
{
    ///sscanf(argv[0], "%s", nomeArquivo);
    //printf("Teste: %s",argv[1]);
    printf("DES SEQ Tests\n");
    enc_dec_file(argv[1]);
    // printf("DES test: %s\n", des_test() ? "SUCCEEDED" : "FAILED");
    return(0);
}
