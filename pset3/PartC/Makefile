# Part-C: Convolution in CUDA
NVCC = nvcc
NVCCFLAGS = -O2 -std=c++11
CUDA_LIBS = -lcudart
CUDNN_LIBS = -lcudnn

script: 
	chmod +x *.sh
all: c1 c2 c3 script

# c1: Simple convolution without tiling or shared memory
c1: c1.cu timer.cu
	$(NVCC) $(NVCCFLAGS) -o c1 c1.cu timer.cu $(CUDA_LIBS)

# c2: Tiled convolution with shared memory (to be implemented)
c2: c2.cu timer.cu
	$(NVCC) $(NVCCFLAGS) -o c2 c2.cu timer.cu $(CUDA_LIBS)

# c3: Convolution with cuDNN
c3: c3.cu timer.cu
	$(NVCC) $(NVCCFLAGS) -o c3 c3.cu timer.cu $(CUDA_LIBS) $(CUDNN_LIBS)

# clean all executables
clean:
	rm -f c1 c2 c3

