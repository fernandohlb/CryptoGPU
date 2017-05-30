#! /bin/bash

set -o xtrace

ITERATIONS=3
THREADS=256

# #AES: Sequential
# mkdir -p AES/results_perf/aes_seq

# for ((i=1; i<=$ITERATIONS; i++)); do
#     for file in sample_files/*; do
#         ./AES/seq/aes $file>> AES/results_perf/aes_seq/$(basename $file | cut -d'.' -f1).log 2>&1
#     done
# done

# #AES: GPU
# mkdir -p AES/results_perf/aes_gpu

# for ((i=1; i<=$ITERATIONS; i++)); do
#     for file in sample_files/*; do
#         optirun ./AES/gpu/aes $file>> AES/results_perf/aes_gpu/$(basename $file | cut -d'.' -f1).log 2>&1
#     done
# done

#DES: Sequential
mkdir -p DES/results_perf/des_seq

for ((i=1; i<=$ITERATIONS; i++)); do
    for file in sample_files/*; do
        ./DES/seq/des $file>> DES/results_perf/des_seq/$(basename $file | cut -d'.' -f1).log 2>&1
    done
done

#DES: GPU
mkdir -p DES/results_perf/des_gpu

for ((i=1; i<=$ITERATIONS; i++)); do
    for file in sample_files/*; do
        optirun ./DES/gpu/des $file $THREADS>> DES/results_perf/des_gpu/$(basename $file | cut -d'.' -f1).log 2>&1
    done
done

# #ROT-13: Sequential
# mkdir -p ROT13/results_perf/rot13_seq

# for ((i=1; i<=$ITERATIONS; i++)); do
#     for file in sample_files/*; do
#         ./ROT13/seq/rot13 $file>> ROT13/results_perf/rot13_seq/$(basename $file | cut -d'.' -f1).log 2>&1
#     done
# done

# #ROT-13: GPU
# mkdir -p ROT13/results_perf/rot13_gpu

# for ((i=1; i<=$ITERATIONS; i++)); do
#     for file in sample_files/*; do
#         optirun ./ROT13/gpu/rot13 $file $THREADS>> ROT13/results_perf/rot13_gpu/$(basename $file | cut -d'.' -f1).log 2>&1
#     done
# done

############################################################################################################
# mv *.log DES/results_perf/des_seq

#DES: GPU 
# mkdir -p DES/results_perf/des_gpu

# for ((i=1; i<=$ITERATIONS; i++)); do
   
#     perf stat ./DES/gpu/des_exec sample_files/hubble_1.tif $THREADS>> hubble_1.log 2>&1
#     perf stat ./DES/gpu/des_exec sample_files/hubble_2.tif $THREADS>> hubble_2.log 2>&1
#     perf stat ./DES/gpu/des_exec sample_files/hubble_3.tif $THREADS>> hubble_3.log 2>&1
#     perf stat ./DES/gpu/des_exec sample_files/hubble_4.tif $THREADS>> hubble_4.log 2>&1
# done

# mv *.log DES/results_perf/des_gpu

# #ROT-13: Sequential 
# mkdir -p ROT13/results_perf/rot13_seq

# for ((i=1; i<=$ITERATIONS; i++)); do
    
#     perf stat ./ROT13/seq/rot-13 sample_files/texto_1.txt>> texto_1.log 2>&1
#     perf stat ./ROT13/seq/rot-13 sample_files/texto_3.txt>> texto_3.log 2>&1
#     perf stat ./ROT13/seq/rot-13 sample_files/texto_4.txt>> texto_4.log 2>&1
#     perf stat ./ROT13/seq/rot-13 sample_files/texto_5.txt>> texto_5.log 2>&1
# done

# mv *.log ROT13/results_perf/rot13_seq

# #ROT-13: GPU 
# mkdir -p ROT13/results_perf/rot13_gpu
# #SIZE=$INITIAL_SIZE
# for ((i=1; i<=$ITERATIONS; i++)); do
    
#     perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_1.txt $THREADS>> texto_1.log 2>&1
#     perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_3.txt $THREADS>> texto_3.log 2>&1
#     perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_4.txt $THREADS>> texto_4.log 2>&1
#     perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_5.txt $THREADS>> texto_5.log 2>&1
# done

# mv *.log ROT13/results_perf/rot13_gpu
