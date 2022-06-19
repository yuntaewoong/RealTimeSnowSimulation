/*
	바닥과의 마찰력을 계산
*/

#include "Common.cuh"
#include "SnowCuda.h"

__global__
void friction(float3* dev_pos, float3* dev_vel, float3* dev_force,ParticleData* dev_pData, float3* debug , unsigned int size ,ParamSet Param) {
	int stride = gridDim.x * blockDim.x;
	float3 friction_Force;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx <  size; idx += stride) {
		friction_Force = make_float3(0.0f, 0.0f, 0.0f);
		float3 negativeVelocity = make_float3(-dev_vel[idx].x, -dev_vel[idx].y, 0.f);
		float3 centerVector = make_float3(0.f, 0.f, 0.f);
		float dist = getDistance(negativeVelocity, centerVector);
		if (dev_pos[idx].z <= Param.startFloor.z + dev_pData[idx].radius * 1.2) {//땅에 붙었다면
			float3 friction_direction = make_float3(-dev_vel[idx].x, -dev_vel[idx].y, 0.f) / dist;
			friction_Force = Param.coefficient_friction * Param.particle_mass * Param.gravity * friction_direction;//마찰력 = 수직항력 * 마찰계수 * 속도반대방향벡터
			if (dist < 0.01f)//속도가 아주 낮을때는 무의미한 마찰력 계산 x
			{
				dev_vel[idx].x = 0.f;
				dev_vel[idx].y = 0.f;
				friction_Force = make_float3(0.f, 0.f, 0.f);
			}
		}
		dev_force[idx] = dev_force[idx] + friction_Force;
	}
}