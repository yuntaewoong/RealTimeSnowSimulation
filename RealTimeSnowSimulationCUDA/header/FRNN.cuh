#include "Common.cuh"
#include "SnowCuda.h"


__global__
void resetPBM(int* dev_PBM, int* dev_Count, const unsigned int BinLen) {
	int stride = gridDim.x * blockDim.x;
	for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < BinLen + 1; idx += stride) {
		dev_Count[idx] = 0;
		dev_PBM[idx] = 0;
	}
}

__global__
void countElem(int* dev_BinID, int* dev_Count, const unsigned int size) {
    int stride = gridDim.x * blockDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        atomicAdd(&dev_Count[dev_BinID[idx]], 1);

    }
}

__global__
void getBinLoc(const float3* dev_pos, int* dev_BinID, int* dev_PID, const unsigned int size, const ParamSet Param) {

    int stride = gridDim.x * blockDim.x;
    int3 coord3;

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        coord3.x = floor((dev_pos[idx].x - Param.startFloor.x) / Param.h);
        coord3.y = floor((dev_pos[idx].y - Param.startFloor.y) / Param.h);
        coord3.z = floor((dev_pos[idx].z - Param.startFloor.z) / Param.h);

        dev_PID[idx] = idx;
        dev_BinID[idx] = coord3.x + coord3.y * Param.binSize3.x + coord3.z * Param.binSize3.x * Param.binSize3.y;
    }
}

void SnowCUDA::calcBin(float3 _startFloor, float3 _endTop) {

	Param.startFloor = _startFloor;
	Param.endTop = _endTop;
	Param.binSize3.x = ceil((_endTop.x - _startFloor.x) / Param.h);
	Param.binSize3.y = ceil((_endTop.y - _startFloor.y) / Param.h);
	Param.binSize3.z = ceil((_endTop.z - _startFloor.z) / Param.h);

	Param.binLen = Param.binSize3.x * Param.binSize3.y * Param.binSize3.z;
}