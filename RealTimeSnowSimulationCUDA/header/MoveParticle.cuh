/*==========================================================
File Name: (구)MoveParticle.cuh -> (구)  Acceleration.cuh -> (현)MoveParticle.cuh
각 파티클의 합력을 바탕으로 가속도를 계산
파티클의 가속도를 바탕으로 속도를 계산
속도를 파탕으로 위치를 계산

===========================================================*/
#pragma once
#include "Common.cuh"
#include "SnowCuda.h"

__global__
void moveParticles(float3* dev_pos, float3* dev_vel,float3* dev_force, ParticleData* dev_pData, const float dt, float3* debug,const unsigned int size ,const ParamSet Param) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        float3 accel = dev_force[idx] / Param.particle_mass;
        
        dev_vel[idx] = dev_vel[idx] + dt * accel;

        if (dev_vel[idx].x > Param.maxSpeed) {
            dev_vel[idx].x = Param.maxSpeed;
        }
        else if (dev_vel[idx].x < -Param.maxSpeed) {
            dev_vel[idx].x = -Param.maxSpeed;
        }
        if (dev_vel[idx].y > Param.maxSpeed) {
            dev_vel[idx].y = Param.maxSpeed;
        }
        else if (dev_vel[idx].y < -Param.maxSpeed) {
            dev_vel[idx].y = -Param.maxSpeed;
        }
        if (dev_vel[idx].z > Param.maxSpeed) {
            dev_vel[idx].z = Param.maxSpeed;
        }
        else if (dev_vel[idx].z < -Param.maxSpeed) {
            dev_vel[idx].z = -Param.maxSpeed;
        }

        dev_pos[idx] = dev_pos[idx] + dt * dev_vel[idx];

        // 경계조건
        if (dev_pos[idx].x < Param.startFloor.x + dev_pData[idx].radius) {
            dev_pos[idx].x = Param.startFloor.x + dev_pData[idx].radius;
            AddForceForCompression(make_float3(- dev_vel[idx].x * Param.particle_mass/dt,0,0 ), dev_pData, idx);
            dev_vel[idx].x = 0;
        }
        else if (dev_pos[idx].x > Param.endTop.x - dev_pData[idx].radius) {
            dev_pos[idx].x = Param.endTop.x - dev_pData[idx].radius;
            AddForceForCompression(make_float3(- dev_vel[idx].x * Param.particle_mass / dt, 0, 0), dev_pData, idx);
            dev_vel[idx].x = 0;
        }
        if (dev_pos[idx].y < Param.startFloor.y + dev_pData[idx].radius) {
            dev_pos[idx].y = Param.startFloor.y + dev_pData[idx].radius;
            AddForceForCompression(make_float3(0, - dev_vel[idx].y * Param.particle_mass / dt, 0), dev_pData, idx);
            dev_vel[idx].y = 0;
        }
        else if (dev_pos[idx].y > Param.endTop.y - dev_pData[idx].radius) {
            dev_pos[idx].y = Param.endTop.y - dev_pData[idx].radius;
            AddForceForCompression(make_float3(0, - dev_vel[idx].y * Param.particle_mass / dt, 0), dev_pData, idx);
            dev_vel[idx].y = 0;
        }
        if (dev_pos[idx].z < Param.startFloor.z + dev_pData[idx].radius) {
            dev_pos[idx].z = Param.startFloor.z + dev_pData[idx].radius;
            AddForceForCompression(make_float3(0, 0, - dev_vel[idx].z * Param.particle_mass / dt), dev_pData, idx);
            dev_vel[idx].z = 0;
        }
        else if (dev_pos[idx].z > Param.endTop.z - dev_pData[idx].radius) {
            dev_pos[idx].z = Param.endTop.z - dev_pData[idx].radius;
            AddForceForCompression(make_float3(0, 0, - dev_vel[idx].z * Param.particle_mass / dt), dev_pData, idx);
            dev_vel[idx].z = 0;
        }
    }
}