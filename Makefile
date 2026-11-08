.DEFAULT_GOAL := qwen3moe
CXX        ?= g++
CXXFLAGS   ?= -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS    ?=
NVCC       ?= nvcc
NVCC_CCBIN ?= /usr/bin/g++-15
CUDA_OBJ   :=
# GPU build: make CUDA=1 CUDA_HOME=/usr/local/cuda
# (switching CUDA on/off needs `make clean` first)
ifdef CUDA
CUDA_HOME ?= /usr/local/cuda
CXXFLAGS  += -DEAHMMF_CUDA -DEAHMMF_CUDA_COMPUTE -I$(CUDA_HOME)/include
LDFLAGS   += -L$(CUDA_HOME)/lib64 -lcudart
CUDA_OBJ  := expert_cuda.o
endif

HDRS = eahmmf.h unicode_tables.h qwen3moe.cpp

expert_cuda.o: expert_cuda.cu
	$(NVCC) -ccbin $(NVCC_CCBIN) -O2 -std=c++17 -c $< -o $@

qwen3moe: qwen3moe.cpp eahmmf.h unicode_tables.h $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ qwen3moe.cpp $(CUDA_OBJ) $(LDFLAGS)

test_eahmmf: tests/test_eahmmf.cpp $(HDRS) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_eahmmf.cpp $(CUDA_OBJ) $(LDFLAGS)

unicode_tables.h: gen_unicode.py
	python3 gen_unicode.py

# full verification: numeric reference, pre-tokenizer, baseline equivalence, residency suite
check: qwen3moe test_eahmmf
	python3 tests/test_tiny.py
	python3 tests/test_pretok.py
	sh check_baseline.sh
	./test_eahmmf tiny.gguf

clean:
	rm -f qwen3moe test_eahmmf expert_cuda.o
