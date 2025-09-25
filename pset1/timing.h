#include <time.h>
#include <stdio.h>
#include <stdlib.h> 

static inline double now_sec(){
	struct timescpe time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	return time.tv_sec + time.tv_nsec * 1e-9;
}
