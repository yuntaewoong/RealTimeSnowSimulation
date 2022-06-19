/*===========================================================================
File name: Thermodynamics.cuh
Summary: compute Thermodynamics
		 만약 amount of water의 값이 1에 도달하면 해당 파티클은 물이 된다.
============================================================================*/
#pragma once
#include "Common.cuh"
#include "SnowCuda.h"
#include <math.h>
#define PI 3.14f
/*===========================================================================
Function: computeThermodynamics
Summary: compute Thermodynamics and change phase
Args: float3* dev_pos
		position of particles
	  float* dev_temperature
		temperature of particles
	  float* dev_amountOfWater
		amount of water of particles. save latent heat
	  ParticleData* dev_pData
		data of particles
	  const unsigned int size
	    size
	  const ParamSet Param
	    Param
	  float deltaTime
		deltaTime
	  float tempOfAir
	    temperature of the Air. kelvin degree(K)
	  float tempOfAir
		temperature of the Ground. kelvin degree(K)
	  float effectiveRadius
============================================================================*/
__global__
void computeThermodynamics(float3* dev_pos, float* dev_temperature, float* dev_amountOfWater, ParticleData* dev_pData, const unsigned int size, const ParamSet Param, float deltaTime, float tempOfAir, float tempOfGround, float effectiveRadius = 0.11)
{
	float dist = 0.0f;
	float deltaTemp = 0.0f;
	float qNeighbors = 0.0f;
	float qAir = 0.0f;
	float qGround = 0.0f;
	float qf = 0.0f;
	float qTotal = 0.0f;
	float areaOfParticle = 0.0f;
	int numOfNeighboringParticle = 0;

	int stride = gridDim.x * blockDim.x;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride)
	{
		// 이미 물이면 계산X
		if (dev_amountOfWater[idx] >= 1.0f)
		{
			continue;
		}

		// 1. compute Heat transfer between sph particles
		numOfNeighboringParticle = 0;
		deltaTemp = 0.0f;
		for (int j = 0; j < size; j++)
		{
			if (idx == j)
			{
				continue;
			}
			// check distance between particle idx and j
			dist = getDistance(dev_pos[idx], dev_pos[j]);

			// including only dist < support radius from particle i || dist < 1.1(effective radius)
			// support radius = 0.15
			if (dist < 0.15 || dist < effectiveRadius)
			{
				// calculate deltatemp = (Tj - Ti)/Pj * 45 * (Hh - dist)/(PI * Hh^6)
				deltaTemp += (dev_temperature[j] - dev_temperature[idx]) / ((Param.particle_mass)/(4*PI*pow(dev_pData[j].radius, 3)/3)) * (45 * (effectiveRadius - dist) / (PI * pow(effectiveRadius, 6)));
			}
			// if particles touch each other
			if (dist <= dev_pData[idx].radius * 2)
			{
				numOfNeighboringParticle += 1;
			}
		}
		deltaTemp = 0.5 * Param.particle_mass * deltaTemp * deltaTime;

		// 2. compute heat loss/gain with the neighboring particles
		qNeighbors = ((1 - dev_pData[idx].phase_snow) *Param.hearCap_ice + dev_pData[idx].phase_snow * Param.heatCap_snow) * Param.particle_mass * deltaTemp;

		// 3. compute the heat exchange between the snow particle i and the air and the ground
		// 1) air
		if (numOfNeighboringParticle > 6)
		{
			areaOfParticle = 0;
			qAir = 0;
		}
		else
		{
			areaOfParticle = ((6 - numOfNeighboringParticle) / 6) * (4 * PI * pow(dev_pData[idx].radius, 2));
			qAir = ((1 - dev_pData[idx].phase_snow) * Param.ther_ice + dev_pData[idx].phase_snow * Param.ther_snow) * (tempOfAir - dev_temperature[idx]) * areaOfParticle;
		}
		// 2) ground 
		qGround = 0; // 우선 0으로 지정...
		/*
		if (numOfNeighboringParticle > 6)
		{
			areaOfParticle = 0;
			qGround = 0;
		}
		else
		{
			areaOfParticle = ((6 - numOfNeighboringParticle) / 6) * (4 * PI * pow(dev_pData[idx].radius, 2));
			qGround = ((1 - dev_pData[idx].phase_snow) * Param.ther_ice + dev_pData[idx].phase_snow * Param.ther_snow) * (tempOfGround - dev_temperature[idx]) * areaOfParticle;
		}
		*/
		
		// 4. compute the latent heat
		qf = Param.particle_mass * 334.0f * dev_amountOfWater[idx];

		// 5. compute the total heat value received for a particle
		qTotal = qNeighbors + qGround + qAir + qf;

		// 6. calculate temperature transition
		// 온도 변화량 계산
		deltaTemp = qTotal / (((1 - dev_pData[idx].phase_snow) * Param.hearCap_ice + dev_pData[idx].phase_snow * Param.heatCap_snow) * Param.particle_mass);
		if (dev_temperature[idx] + deltaTemp >= 273.0f)
		{
			// 파티클의 온도가 녹는점에 도달한 경우
			deltaTemp = dev_temperature[idx] + deltaTemp - 273.0f; //잠열 계산에 사용함
			dev_temperature[idx] = 273.0f;

			// 잠열 계산
			dev_amountOfWater[idx] = deltaTemp * ((1 - dev_pData[idx].phase_snow) * Param.hearCap_ice + dev_pData[idx].phase_snow * Param.heatCap_snow) * Param.particle_mass / 80;
		}
		else
		{
			// 온도가 아직 녹는점에 도달하지 못함
			dev_temperature[idx] += deltaTemp;
		}
	}
}