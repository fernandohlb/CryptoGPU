#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <math.h>
#include <time.h>
#include <stdio.h>

#define STEPS_SIZE 3

enum step {alloc, calc, ioops};

void start_timers(double cpu_start[], struct timespec wc_start[], enum step s);
void end_timers(double cpu_end[], struct timespec wc_end[], enum step s);
void print_elapsed(double cpu_start[], struct timespec wc_start[], double cpu_end[], struct timespec wc_end[]);
void print_time(double cpu_start[], struct timespec wc_start[], double cpu_end[], struct timespec wc_end[]);

#endif
