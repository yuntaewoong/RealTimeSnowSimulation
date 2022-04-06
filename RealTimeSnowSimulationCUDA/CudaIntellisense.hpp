/*----------------------------------------------------------
  File:      CudaIntellisense.hpp
  Summary:   Define macros for Visual Studio Intellisense not to catch unnecessary syntax error
				
				ex/ kernel<<<grid,block>>>(a,b,c)
					__global__
					__host__
------------------------------------------------------------*/




#pragma once
#ifdef __INTELLISENSE__
// Only Valid On Intellisense Compiler

//KERNEL_ARG2(grid, block) : <<< grid, block >>>
#define KERNEL_ARG2(grid, block)
//KERNEL_ARG3(grid, block, sh_mem) : <<< grid, block, sh_mem >>>
#define KERNEL_ARG3(grid, block, sh_mem)
//KERNEL_ARG4(grid, block, sh_mem, stream) : <<< grid, block, sh_mem, stream >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream)

// __CUDACC__ 가 define 된 것"처럼" 보여주기
#define __CUDACC__

// CUDA 관련 매크로들을 공백으로 재정의한 것"처럼" 보여주기
#define __global__ 
#define __host__ 
#define __device__ 
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#define __constant__ 
#define __shared__ 
#define __restrict__
#define __noinline__
#define __forceinline__
#define __managed__
#else
//Valid On Real Compiler

#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif