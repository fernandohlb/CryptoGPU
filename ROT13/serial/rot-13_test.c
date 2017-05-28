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
#include "rot-13.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rot13_test()
{
    char *data = NULL; 
    char *encrypted_data; 
    char *decrypted_data;  
    int pass = 1;
    const char *filename = "sample_files/king_james_bible.txt";
    struct stat st;
 
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

    encrypted_data = (char *) malloc(sizeof(char) * st.st_size);
    decrypted_data = (char *) malloc(sizeof(char) * st.st_size);

    char * data_buf = malloc(sizeof(char) * st.st_size);

    // To encode, just apply ROT-13.
    strcpy(data_buf, data);
    
    rot13(data_buf);
    strcpy(encrypted_data, data_buf);
    //pass = pass && !strcmp(code, buf);

    // To decode, just re-apply ROT-13.
    rot13(data_buf);
    strcpy(decrypted_data, data_buf);

    //Compara o dado Decriptografado com o Dado Original
    pass = pass && !strcmp(decrypted_data, data);

    FILE *enc_file = fopen("king_james_bible_enc.txt", "wb+");
    FILE *dec_file = fopen("king_james_bible_dec.txt", "wb+");
    
    fwrite(encrypted_data, sizeof(char) * st.st_size, 1, enc_file);
    fwrite(decrypted_data, sizeof(char) * st.st_size, 1, dec_file);
    return(pass);
}

int main()
{
    printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");

    return(0);
}
