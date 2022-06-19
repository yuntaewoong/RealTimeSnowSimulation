#pragma once
/*==========================================================
File Name: Compression.cuh

각 입자당 계산한 합력에 근거해서 각 파티클의 d값, 반지름 값을 업데이트

미리 계산되어야 하는것: 각 파티클의 positiveForce, negativeForce
실행 결과: 각 파티클의 d,r값 업데이트

===========================================================*/


#include "Common.cuh"
#include "SnowCuda.h"
#include "math_constants.h"



__global__
void compression(ParticleData* dev_pData, float* dev_radius,float3* debug, const unsigned int size,const ParamSet Param) {
    int stride = gridDim.x * blockDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        
        //한 입자에 작용하는 4개의 힘(중력,전단력,약력,응집력)의 x,y,z축으로의 양의방향의 힘, 음의 방향의 힘의 합을 구함
        

        // compression (물체가 찌그러지는데 영향을 주는 힘은 힘의 크기의 절댓값이 작은힘임(ex/ x+ 방향으로 3N의 힘, x-방향으로 1N의 힘을 받으면, 물체의 합력은 2N으로 양의방향으로 이동하지만, 물체자체는 1N의 힘만큼 찌그러지는 영향을 받게됨)
        float minXSquare = fminf(dev_pData[idx].positiveForce.x * dev_pData[idx].positiveForce.x, dev_pData[idx].negativeForce.x * dev_pData[idx].negativeForce.x);
        float minYSquare = fminf(dev_pData[idx].positiveForce.y * dev_pData[idx].positiveForce.y, dev_pData[idx].negativeForce.y * dev_pData[idx].negativeForce.y);
        float minZSquare = fminf(dev_pData[idx].positiveForce.z * dev_pData[idx].positiveForce.z, dev_pData[idx].negativeForce.x * dev_pData[idx].negativeForce.x);

        float compressiveForces = sqrtf(minXSquare + minYSquare + minZSquare);//압력 계산

        float p_c = compressiveForces / (CUDART_PI_F * dev_pData[idx].radius * dev_pData[idx].radius);//압력을 파티클 면적으로 나눔(압력 = 단위면적당 받는 힘 N/m^2)

        float pi = 100.0f * dev_pData[idx].phase_snow + 900.0f * (1 - dev_pData[idx].phase_snow);
        float e = expf(1.0f);//자연상수 e
        float Dpi = Param.minForceW + Param.maxForceW * ((powf(e, (pi / 100.f - 1)) - 0.000335f) / 2980.96f);//논문 저자의 식 적용

        // update radius
        if (compressiveForces > Dpi)
        {
            dev_pData[idx].d = fmaxf(dev_pData[idx].d - Param.k_q * p_c, 0);

            dev_pData[idx].radius = dev_pData[idx].d * Param.radius_snow + (1 - dev_pData[idx].d) * Param.radius_ice;
            dev_pData[idx].phase_snow = dev_pData[idx].d;
        }
        dev_radius[idx] = dev_pData[idx].radius;
    }

}