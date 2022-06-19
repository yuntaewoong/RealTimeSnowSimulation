#pragma once
#include "Common.cuh"
#include "SnowCuda.h"


__global__
void initParticles(ParticleData* dev_pData, float3* dev_velocity, float* dev_radius, float phase_snow, const unsigned int size, ParamSet Param) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
            dev_pData[idx].phase_snow = phase_snow;
            dev_pData[idx].radius = phase_snow * Param.radius_snow + (1 - phase_snow) * Param.radius_ice;
            dev_pData[idx].d = phase_snow;
            dev_velocity[idx] = make_float3(0.f, 0.f, 0.f);
            dev_radius[idx] = dev_pData[idx].radius;
    }
}


__global__
void initNeighbors(float3* dev_pos,ParticleData* dev_pData, const unsigned int size) {
    int stride = gridDim.x * blockDim.x;
    int count;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        for (int i = 0; i < 27; i++) {
            dev_pData[idx].CohNeighbor[i] = -1;
        }
        count = 0;
        for (int j = 0; j < size; j++) {
            if (idx == j) continue;
            if (getDistance(dev_pos[idx], dev_pos[j]) <= (dev_pData[idx].radius + dev_pData[j].radius) * 1.75f) {
                if (count < 27) {
                    dev_pData[idx].CohNeighbor[count] = j;
                    count++;
                }
            }

        }
    }
}