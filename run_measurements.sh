#! /bin/bash

set -o xtrace

MEASUREMENTS=10
ITERATIONS=10
THREADS=256

#DES: Sequential
mkdir -p DES/results_perf/des_seq
#SIZE=$INITIAL_SIZE
for ((i=1; i<=$ITERATIONS; i++)); do
    
    perf stat ./DES/seq/des "sample_files/hubble_1.tif">> hubble_1.log 2>&1
    perf stat ./DES/seq/des "sample_files/hubble_2.tif">> hubble_2.log 2>&1
    perf stat ./DES/seq/des "sample_files/hubble_3.tif">> hubble_3.log 2>&1
    perf stat ./DES/seq/des "sample_files/hubble_4.tif">> hubble_4.log 2>&1
done

mv *.log DES/results_perf/des_seq

##DES: GPU 
#mkdir -p DES/results_perf/des_gpu
##SIZE=$INITIAL_SIZE
#for ((i=1; i<=$ITERATIONS; i++)); do
    
#    perf stat ./DES/gpu/des_exec sample_files/hubble_1.tif $THREADS>> hubble_1.log 2>&1
#    perf stat ./DES/gpu/des_exec sample_files/hubble_2.tif $THREADS>> hubble_2.log 2>&1
#    perf stat ./DES/gpu/des_exec sample_files/hubble_3.tif $THREADS>> hubble_3.log 2>&1
#    perf stat ./DES/gpu/des_exec sample_files/hubble_4.tif $THREADS>> hubble_4.log 2>&1
#done

#mv *.log DES/results_perf/des_gpu

##ROT-13: Sequential 
#mkdir -p ROT13/results_perf/rot13_seq
##SIZE=$INITIAL_SIZE
#for ((i=1; i<=$ITERATIONS; i++)); do
    
#    perf stat ./ROT13/seq/rot-13 sample_files/texto_1.txt>> texto_1.log 2>&1
#    perf stat ./ROT13/seq/rot-13 sample_files/texto_3.txt>> texto_3.log 2>&1
#    perf stat ./ROT13/seq/rot-13 sample_files/texto_4.txt>> texto_4.log 2>&1
#    perf stat ./ROT13/seq/rot-13 sample_files/texto_5.txt>> texto_5.log 2>&1
#done

#mv *.log ROT13/results_perf/rot13_seq

##ROT-13: GPU 
#mkdir -p ROT13/results_perf/rot13_gpu
##SIZE=$INITIAL_SIZE
#for ((i=1; i<=$ITERATIONS; i++)); do
    
#    perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_1.txt $THREADS>> texto_1.log 2>&1
#    perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_3.txt $THREADS>> texto_3.log 2>&1
#    perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_4.txt $THREADS>> texto_4.log 2>&1
#    perf stat ./ROT13/gpu/rot-13_exec sample_files/texto_5.txt $THREADS>> texto_5.log 2>&1
#done

#mv *.log ROT13/results_perf/rot13_gpu
