/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
//#include<iostream>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

__device__ void tile_partition(cooperative_groups::thread_group g)
{
cooperative_groups::thread_group tile8 = cooperative_groups::tiled_partition(g,16);
//printf("tile8 size is: %d\n",tile8.size());
size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
//printf("offset: %d\n",offset);
printf("thread rank is: %d\n",tile8.thread_rank());

if(offset<8)
//if(tile8.thread_rank() <8)
{
//__syncthreads();
tile8.sync();
printf("I am after tile8.sync()\n");
printf("I am in offset<8\n");
}
else if((offset>7) && (offset<16))
{

printf("I am in offset<16\n");
}
else if((offset>15) && (offset<24))
{

printf("I am in offset<24\n");
}


tile8.sync();
//__syncthreads();
}




__global__ void
//vector_square(float *C_d, float *A_d, size_t N)
thread_partition()
{
/*    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    unsigned int rank = grid.thread_rank();
    unsigned int grid_size = grid.size();*/
    cooperative_groups::thread_group g = cooperative_groups::this_thread_block();
    tile_partition(g);
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);

}


int main(int argc, char *argv[])
{
    CHECK(cudaSetDevice(2));
  //  float *A_d, *C_d;
   // float *A_h, *C_h;
    //size_t N = 1000000;
    size_t N = 32;
    size_t Nbytes = N * sizeof(float);

    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);

/*    printf ("info: copy Host2Device\n");
    CHECK ( cudaMemcpy(A_d, A_h, Nbytes, cudaMemcpyHostToDevice));
 int max_blocks_per_sm;
    CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
            vector_square, 32, 0));*/

    const unsigned threadsPerBlock = 64;
    const unsigned blocks = N/threadsPerBlock;

    printf ("info: launch 'vector_square' kernel\n");
//    vector_square <<<blocks, threadsPerBlock>>> (C_d, A_d, N);
//CHECK(cudaDeviceSynchronize());
void *coop_params=NULL;


/*coop_params[0]=(void*)&C_d,
coop_params[1]=(void*)&A_d;
coop_params[2]=(void*)&N;
cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));*/
cudaError_t errval=(cudaLaunchCooperativeKernel((void*)thread_partition,blocks,threadsPerBlock,&coop_params,0,0));
//cudaError_t errval=(cudaLaunchCooperativeKernel((void*)thread_partition,blocks,threadsPerBlock,NULL,0,0));
CHECK(cudaDeviceSynchronize());
	std::cout<<"errval: "<<cudaGetErrorString(errval)<<std::endl;
    if (errval != cudaSuccess) 
    {
        std::cout << "CUDA error: " << cudaGetErrorString(errval);
        std::cout << std::endl;
        std::cout << "    Location: " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(errval);
    }


    printf ("DONE!\n");
return 0;
}
