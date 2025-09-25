# Makefile

CC = gcc
CFLAGS = -O3 -march=native -Wall

# Targets
all: dp1 dp2

dp1: dp1.c
	$(CC) $(CFLAGS) -o dp1 dp1.c

dp2: dp2.c
	$(CC) $(CFLAGS) -o dp2 dp2.c

clean:
	rm -f dp1 dp2
