# part-B: vector addition on CPU and GPU
# makes doing the experiments a bit easiers
CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++11
NVCCFLAGS = -O2 -std=c++11
CUDA_LIBS = -lcudart

all: q1 q2 q3

# Q1: CPU version
q1: q1.cpp timer.cpp
	$(CXX) $(CXXFLAGS) -o q1 q1.cpp timer.cpp
	chmod +x q1

# Q2: CUDA version with explicit memory management
q2: q2.cu timer.cpp
	$(NVCC) $(NVCCFLAGS) -o q2 q2.cu timer.cpp $(CUDA_LIBS)
	chmod +x q2

# Q3: CUDA version with unified memory
q3: q3.cu timer.cpp
	$(NVCC) $(NVCCFLAGS) -o q3 q3.cu timer.cpp $(CUDA_LIBS)
	chmod +x q3

# make scripts executable
scripts:
	chmod +x *.sh

# clean all executables and object files
clean:
	rm -f q1 q2 q3
