###
### For COMS E6998 Spring 2024
### Instructor: Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###


# paths and setup
SDK_INSTALL_PATH := /usr/local/cuda/
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := fbg2107CUDA1.tar
EXECS :=  vecadd00 vecadd01 #matmult00
SCRIPTS := run_multiple_trials_and_save.sh

all:$(EXECS)
	chmod +x *.sh

#######################################################################
clean:
	rm -f $(EXECS) *.o 
	rm ./results/*.txt

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################
# compile the timer to be used in different programs
timer.o : timer.cu timer.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
# original non-coalesced vector addition (Q1)
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

# dependencies for timing, note we are using vecadd.cu as main program
# but vecaddKernel00.o as the kernel object file, this can change as 
# as we see below, where we write a different kernel 
vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o 
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
# optimized coalesced vector addition (Q2)

vecaddKernel01.o : vecaddKernel01.cu # notice, we use vecaddKernel01.o now
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o 
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
# make scripts executable
scripts:
	chmod +x *.sh


#######################################################################

# matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
# 	${NVCC} $< -c -o $@ $(OPTIONS)

# matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
# 	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
### matmultKernel01.o : matmultKernel01.cu matmultKernel.h
###	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32
###
### matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
###	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32




