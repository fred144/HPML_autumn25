# Makefile for pset1
CFLAGS = -O3 -Wall

all: dp1 dp2 dp3

dp1: dp1.c
	gcc $(CFLAGS) -o dp1 dp1.c

dp2: dp2.c
	gcc $(CFLAGS) -o dp2 dp2.c

dp3: dp3.c
	gcc $(CFLAGS) -o dp3 dp3.c \
	    -I${MKLROOT}/include \
	    -L${MKLROOT}/lib/intel64 \
    -lmkl_rt -lpthread -lm -ldl

clean:
	rm -f dp1 dp2 dp3
