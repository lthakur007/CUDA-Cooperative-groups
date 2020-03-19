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


/* 
 * Square each element in the array A and write to array C.
 */
__global__ void
vector_square()
{
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    unsigned int rank = grid.thread_rank();
    unsigned int grid_size = grid.size();

    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);

//	printf("grid rank: %d\n",rank);
    if(offset == 0){
	printf("grid.is_valid(): %d\n",grid.is_valid());
	printf("grid_size: %d\n",grid_size);
	//printf("grid rank: %d\n",rank);
	//dim3 thread_loc=grid.group_index();//group_index() is not a member in cooperative group
	}

/*if(offset == 0)
printf("I am from thread 0\n");
else if(offset == 40)
printf("I am from thread 40 \n");
if(offset==40){
//__syncthreads();
unsigned long long int wait_t=32000000000,start=clock64(),cur;
    do{cur=clock64()-start;}
    while(cur<wait_t);
printf("Wait is over!\n");
}
grid.sync();
if(offset == 0)
printf("I am after grid.sync() from thread 0\n");
else if(offset == 40)
printf("I am after grid.sync() from thread 40 \n");*/
}


int main(int argc, char *argv[])
{
    CHECK(cudaSetDevice(2));
    size_t N = 2560;

    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    int warp_size = props.warpSize;
    int num_sms = props.multiProcessorCount;
    int max_blocks_per_sm;
    // Calculate the device occupancy to know how many blocks can be run.
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &max_blocks_per_sm, vector_square, warp_size, 0,
            cudaOccupancyDefault));

    int desired_blocks = max_blocks_per_sm * num_sms;
    /*std::cout<<"warp_size: "<<warp_size<<std::endl;
    std::cout<<"multiProcessorCount: "<<num_sms<<std::endl;
    std::cout<<"max_blocks_per_sm: "<<max_blocks_per_sm<<std::endl;
    std::cout<<"desired_blocks: "<<desired_blocks<<std::endl;*/


    const unsigned threadsPerBlock = 32;
    const unsigned blocks = N/threadsPerBlock;
    //const unsigned blocks = (desired_blocks+1);

    printf ("info: launch 'vector_square' kernel\n");
//    vector_square <<<blocks, threadsPerBlock>>> (C_d, A_d, N);
//CHECK(cudaDeviceSynchronize());
void *coop_params=NULL;
//cudaError_t errval=(cudaLaunchCooperativeKernel((void*)vector_square,blocks,threadsPerBlock,coop_params,0,0));
cudaError_t errval=(cudaLaunchCooperativeKernel((void*)vector_square,blocks,threadsPerBlock,&coop_params,0,0));
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
