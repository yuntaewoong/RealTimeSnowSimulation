/*==========================================================
File Name: CohesionForce.cuh

각 입자에 작용하는 응집력 계산

미리 계산되어야 하는것: 각 파티클의 neighborCounts값
실행 결과: 각 파티클의 coh_Force, positiveForce, negativeForce값 업데이트

===========================================================*/
#pragma once
#include "Common.cuh"
#include "SnowCuda.h"


__global__
void getCohesionForce2(float3* dev_pos, float3* dev_vel, float3* dev_force, ParticleData* dev_pData, const int* dev_PID, const int* dev_BinID, const int* dev_PBM, float3* debug, const unsigned int size, const ParamSet Param) {

    int stride = gridDim.x * blockDim.x;
    //FRNN을 위한 변수
    int PID;
    int BinID;
    int start;
    int end;
    int start_x;
    int end_x;
    int new_y;
    int new_z;
    int j;
    int3 pBinCoord;


    float dist;
    float delta;
    float3 norVec;
    float youngF;
    float cohF;
    float3 nforce;

    float3 coh_Force;

    float3 tangential_Force;


    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        PID = dev_PID[idx];
        BinID = dev_BinID[idx];


        pBinCoord.x = BinID % Param.binSize3.x;
        pBinCoord.y = (BinID % (Param.binSize3.x * Param.binSize3.y)) / Param.binSize3.x;
        pBinCoord.z = BinID / (Param.binSize3.x * Param.binSize3.y);

        dev_pData[PID].positiveForce = make_float3(0.f, 0.f, 0.f);
        dev_pData[PID].negativeForce = make_float3(0.f, 0.f, 0.f);
        coh_Force = make_float3(0.f, 0.f, 0.f);
        tangential_Force = make_float3(0.f, 0.f, 0.f);

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {

                start_x = max(pBinCoord.x - 1, 0);
                end_x = min(pBinCoord.x + 1, Param.binSize3.x - 1);
                new_y = pBinCoord.y + y;
                new_z = pBinCoord.z + z;
                if (new_y >= Param.binSize3.y || new_y < 0) {
                    continue;
                }
                if (new_z >= Param.binSize3.z || new_z < 0) {
                    continue;
                }
                start = start_x + new_y * Param.binSize3.x + new_z * Param.binSize3.x * Param.binSize3.y;
                end = end_x + new_y * Param.binSize3.x + new_z * Param.binSize3.x * Param.binSize3.y;

                for (int idxP = dev_PBM[start]; idxP < dev_PBM[end + 1]; idxP++) {
                    j = dev_PID[idxP];
                    //==========================================여기서부터 계산관련 코드 작성================================================================================
                    if (PID == j) continue;
                    float dist = getDistance(dev_pos[PID], dev_pos[j]);


                    if (dist >= (dev_pData[PID].radius + dev_pData[j].radius))
                    {
                        int ngh;
                        for (ngh = 0; ngh < 27; ngh++) {
                            if (dev_pData[PID].CohNeighbor[ngh] == j) {
                                break;
                            }
                        }
                        
                        if (ngh < 27) {
                            float3 dir = getNormalVec(dev_pos[j], dev_pos[PID]) * (-1.f);
                            float overlapDist = dev_pData[PID].radius + dev_pData[j].radius - dist;

                            float Ei = Param.E_snow * dev_pData[PID].phase_snow + Param.E_ice * (1 - dev_pData[PID].phase_snow);
                            float Ej = Param.E_snow * dev_pData[j].phase_snow + Param.E_ice * (1 - dev_pData[j].phase_snow);

                            float3 forces = (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist * dir;

                            float cohesive_strength_i = Param.coh_snow * dev_pData[PID].phase_snow + Param.coh_ice * (1 - dev_pData[PID].phase_snow);
                            float cohesive_strength_j = Param.coh_snow * dev_pData[j].phase_snow + Param.coh_ice * (1 - dev_pData[j].phase_snow);

                            float condition1 = -1.f * (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist;
                            float condition2 = 4.f * (cohesive_strength_i * dev_pData[PID].radius * dev_pData[PID].radius + cohesive_strength_j * dev_pData[j].radius * dev_pData[j].radius) / 2.f;

                            if (condition1 < condition2)
                                coh_Force = coh_Force + forces;
                            else {
                                coh_Force = coh_Force + make_float3(0.f, 0.f, 0.f);
                                dev_pData[PID].CohNeighbor[ngh] = -1;
                            }
                                
                        }
                    }



                    else if (dist < (dev_pData[PID].radius + dev_pData[j].radius))
                    {
                        float3 dir = getNormalVec(dev_pos[j], dev_pos[PID]) * (-1.f);
                        float overlapDist = dev_pData[PID].radius + dev_pData[j].radius - dist;

                        float Ei = Param.E_snow * dev_pData[PID].phase_snow + Param.E_ice * (1 - dev_pData[PID].phase_snow);
                        float Ej = Param.E_snow * dev_pData[j].phase_snow + Param.E_ice * (1 - dev_pData[j].phase_snow);

                        float3 forces = (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist * dir;

                        float cohesive_strength_i = Param.coh_snow * dev_pData[PID].phase_snow + Param.coh_ice * (1 - dev_pData[PID].phase_snow);
                        float cohesive_strength_j = Param.coh_snow * dev_pData[j].phase_snow + Param.coh_ice * (1 - dev_pData[j].phase_snow);

                        float condition1 = -1.f * (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist;
                        float condition2 = 4.f * (cohesive_strength_i * dev_pData[PID].radius * dev_pData[PID].radius + cohesive_strength_j * dev_pData[j].radius * dev_pData[j].radius) / 2.f;

                        if (condition1 < condition2)
                            coh_Force = coh_Force + forces;
                        else
                            coh_Force = coh_Force + make_float3(0.f, 0.f, 0.f);



                        float3 vi = dev_vel[PID];
                        float3 vj = dev_vel[j];
                        if ((vi - vj).x != 0.f && (vi - vj).y != 0.f && (vi - vj).z != 0.f)
                        {
                            float3 ut = getNormalVec(vj, vi);
                            float3 Ft = ut * getDistance(forces, make_float3(0.f, 0.f, 0.f)) * Param.tan_ph * 1.f;
                            tangential_Force = tangential_Force + Ft;
                        }
                        AddForceForCompression(coh_Force, dev_pData, PID);
                        AddForceForCompression(tangential_Force, dev_pData, PID);
                        //논문에서 cohesive force는 압력계산에서 제외하라고 언급한 부분은
                        //cohesion force자체를 제외하라는 것이 아닌, cohesive(당기는)힘은 제외하라는 뜻
                        //따라서 dist < p.radius + nei.radius인 상황에서는 
                        //압력계산을 한다
                    }
                }
            }
        }
        AddForceForCompression(make_float3(0.f, 0.f, -Param.gravity) * Param.particle_mass, dev_pData, PID);
        dev_force[PID] = coh_Force + tangential_Force + make_float3(0.f, 0.f, -Param.gravity) * Param.particle_mass;
    }
}

__global__
void getCohesionForce(float3* dev_pos, float3* dev_vel, float3* dev_force, ParticleData* dev_pData, const int* dev_PID, const int* dev_BinID, const int* dev_PBM, float3* debug, const unsigned int size, const ParamSet Param) {

    int stride = gridDim.x * blockDim.x;
    //FRNN을 위한 변수
    int PID;
    int BinID;
    int start;
    int end;
    int start_x;
    int end_x;
    int new_y;
    int new_z;
    int j;
    int3 pBinCoord;


    float dist;
    float delta;
    float3 norVec;
    float youngF;
    float cohF;
    float3 nforce;

    float3 coh_Force;
    float3 weak_Force;
    float3 tangential_Force;
    

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        PID = dev_PID[idx];
        BinID = dev_BinID[idx];
        

        pBinCoord.x = BinID % Param.binSize3.x;
        pBinCoord.y = (BinID % (Param.binSize3.x * Param.binSize3.y)) / Param.binSize3.x;
        pBinCoord.z = BinID / (Param.binSize3.x * Param.binSize3.y);

        dev_pData[PID].positiveForce = make_float3(0.f, 0.f, 0.f);
        dev_pData[PID].negativeForce = make_float3(0.f, 0.f, 0.f);
        coh_Force = make_float3(0.f, 0.f, 0.f);
        tangential_Force = make_float3(0.f, 0.f, 0.f);

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {

                start_x = max(pBinCoord.x - 1, 0);
                end_x = min(pBinCoord.x + 1, Param.binSize3.x - 1);
                new_y = pBinCoord.y + y;
                new_z = pBinCoord.z + z;
                if (new_y >= Param.binSize3.y || new_y < 0) {
                    continue;
                }
                if (new_z >= Param.binSize3.z || new_z < 0) {
                    continue;
                }
                start = start_x + new_y * Param.binSize3.x + new_z * Param.binSize3.x * Param.binSize3.y;
                end = end_x + new_y * Param.binSize3.x + new_z * Param.binSize3.x * Param.binSize3.y;

                for (int idxP = dev_PBM[start]; idxP < dev_PBM[end + 1]; idxP++) {
                    j = dev_PID[idxP];
                    //==========================================여기서부터 계산관련 코드 작성================================================================================
                    if (PID == j) continue;
                    float dist = getDistance(dev_pos[PID], dev_pos[j]);
                    

                    if (((float)dev_pData[PID].neighborCounts / Param.MaxNeighbor >= 0.75 ||
                        (float)dev_pData[j].neighborCounts / Param.MaxNeighbor >= 0.75) &&
                        dist > (dev_pData[PID].radius + dev_pData[j].radius))
                    {
                        float3 dir = getNormalVec(dev_pos[j], dev_pos[PID]) * (-1.f);
                        float overlapDist = dev_pData[PID].radius + dev_pData[j].radius - dist;

                        float Ei = Param.E_snow * dev_pData[PID].phase_snow + Param.E_ice * (1 - dev_pData[PID].phase_snow);
                        float Ej = Param.E_snow * dev_pData[j].phase_snow + Param.E_ice * (1 - dev_pData[j].phase_snow);

                        float3 forces = (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist * dir;

                        float cohesive_strength_i = Param.coh_snow * dev_pData[PID].phase_snow + Param.coh_ice * (1 - dev_pData[PID].phase_snow);
                        float cohesive_strength_j = Param.coh_snow * dev_pData[j].phase_snow + Param.coh_ice * (1 - dev_pData[j].phase_snow);

                        float condition1 = -1.f * (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist;
                        float condition2 = 4.f * (cohesive_strength_i * dev_pData[PID].radius * dev_pData[PID].radius + cohesive_strength_j * dev_pData[j].radius * dev_pData[j].radius) / 2.f;

                        if (condition1 < condition2)
                            coh_Force = coh_Force + forces;
                        else
                            coh_Force = coh_Force + make_float3(0.f, 0.f, 0.f);
                    }
                    else if (dist < (dev_pData[PID].radius + dev_pData[j].radius))
                    {
                        float3 dir = getNormalVec(dev_pos[j], dev_pos[PID]) * (-1.f);
                        float overlapDist = dev_pData[PID].radius + dev_pData[j].radius - dist;

                        float Ei = Param.E_snow * dev_pData[PID].phase_snow + Param.E_ice * (1 - dev_pData[PID].phase_snow);
                        float Ej = Param.E_snow * dev_pData[j].phase_snow + Param.E_ice * (1 - dev_pData[j].phase_snow);

                        float3 forces = (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist * dir;

                        float cohesive_strength_i = Param.coh_snow * dev_pData[PID].phase_snow + Param.coh_ice * (1 - dev_pData[PID].phase_snow);
                        float cohesive_strength_j = Param.coh_snow * dev_pData[j].phase_snow + Param.coh_ice * (1 - dev_pData[j].phase_snow);

                        float condition1 = -1.f * (Ei * dev_pData[PID].radius + Ej * dev_pData[j].radius) / 2.f * overlapDist;
                        float condition2 = 4.f * (cohesive_strength_i * dev_pData[PID].radius * dev_pData[PID].radius + cohesive_strength_j * dev_pData[j].radius * dev_pData[j].radius) / 2.f;

                        if (condition1 < condition2)
                            coh_Force = coh_Force + forces;
                        else
                            coh_Force = coh_Force + make_float3(0.f, 0.f, 0.f);



                        float3 vi = dev_vel[PID];
                        float3 vj = dev_vel[j];
                        if ((vi - vj).x != 0.f && (vi - vj).y != 0.f && (vi - vj).z != 0.f)
                        {
                            float3 ut = getNormalVec(vj, vi);
                            float3 Ft = ut * getDistance(forces, make_float3(0.f, 0.f, 0.f)) * Param.tan_ph * 1.f;
                            tangential_Force = tangential_Force + Ft;
                        }
                        AddForceForCompression(coh_Force, dev_pData, PID);
                        AddForceForCompression(tangential_Force, dev_pData, PID);
                        //논문에서 cohesive force는 압력계산에서 제외하라고 언급한 부분은
                        //cohesion force자체를 제외하라는 것이 아닌, cohesive(당기는)힘은 제외하라는 뜻
                        //따라서 dist < p.radius + nei.radius인 상황에서는 
                        //압력계산을 한다
                    }
                }
            }
        }
        AddForceForCompression(make_float3(0.f, 0.f, -Param.gravity) * Param.particle_mass, dev_pData, PID);
        dev_force[PID] = coh_Force + tangential_Force + make_float3(0.f,0.f, - Param.gravity) * Param.particle_mass;
    }
}