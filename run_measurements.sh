#! /bin/bash

set -o xtrace

MEASUREMENTS=10
ITERATIONS=10

#DES: Sequential
mkdir -p DES/results_perf/des_seq
#SIZE=$INITIAL_SIZE
#for ((i=1; i<=$ITERATIONS; i++)); do
    
#    perf stat ./DES/seq/des "sample_files/hubble_1.tif">> hubble_1.log 2>&1
#    perf stat ./DES/seq/des "sample_files/hubble_2.tif">> hubble_2.log 2>&1
#    perf stat ./DES/seq/des "sample_files/hubble_3.tif">> hubble_3.log 2>&1
#    perf stat ./DES/seq/des "sample_files/hubble_4.tif">> hubble_4.log 2>&1
#done

#mv *.log DES/results_perf/des_seq

#DES: GPU 
mkdir -p DES/results_perf/des_gpu
#SIZE=$INITIAL_SIZE
for ((i=1; i<=$ITERATIONS; i++)); do
    
    perf stat ./DES/gpu/des_exec "sample_files/hubble_1.tif 256">> hubble_1.log 2>&1
    perf stat ./DES/gpu/des_exec "sample_files/hubble_2.tif 256">> hubble_2.log 2>&1
    perf stat ./DES/gpu/des_exec "sample_files/hubble_3.tif 256">> hubble_3.log 2>&1
    perf stat ./DES/gpu/des_exec "sample_files/hubble_4.tif 256">> hubble_4.log 2>&1
done

mv *.log DES/results_perf/des_gpu

#ROT-13: Sequential 
mkdir -p ROT13/results_perf/rot13_seq
#SIZE=$INITIAL_SIZE
for ((i=1; i<=$ITERATIONS; i++)); do
    
    perf stat ./ROT13/seq/rot-13 "sample_files/tale_of_two_cities.txt">> tale_of_two_cities.log 2>&1
    perf stat ./ROT13/seq/rot-13 "sample_files/ulysses.txt">> ulysses.log 2>&1
    perf stat ./ROT13/seq/rot-13 "sample_files/king_james_bible.txt">> king_james_bible.log 2>&1
    perf stat ./ROT13/seq/rot-13 "sample_files/king_james_bible2.txt">> king_james_bible2.log 2>&1
done

mv *.log ROT13/results_perf/rot13_seq

#ROT-13: GPU 
mkdir -p ROT13/results_perf/rot13_gpu
#SIZE=$INITIAL_SIZE
for ((i=1; i<=$ITERATIONS; i++)); do
    
    perf stat ./ROT13/gpu/rot-13_exec "sample_files/tale_of_two_cities.txt">> tale_of_two_cities.log 2>&1
    perf stat ./ROT13/gpu/rot-13_exec "sample_files/ulysses.txt">> ulysses.log 2>&1
    perf stat ./ROT13/gpu/rot-13_exec "sample_files/king_james_bible.txt">> king_james_bible.log 2>&1
    perf stat ./ROT13/gpu/rot-13_exec "sample_files/king_james_bible2.txt">> king_james_bible2.log 2>&1
done

mv *.log ROT13/results_perf/rot13_gpu
