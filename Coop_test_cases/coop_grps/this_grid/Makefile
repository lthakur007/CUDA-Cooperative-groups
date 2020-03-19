CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= $(CXX)
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=this_grid

NVCC_FLAGS=-m64 -lineinfo
SMS = 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += -g -rdc=true

all : ${TARGET}

this_grid.o: this_grid.cu
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

this_grid: this_grid.o
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o

