#pragma once

#include <string>
#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "device_launch_parameters.h"

#define GRID_DIM 32
#define BLOCK_DIM 128
#define MAX_OF_COLLIDER_NOT_INTERACTING 300
#define MAX_OF_COLLIDER_INTERACTING 10
struct ParamSet {
    
    float h; // 완화길이, 주변입자가 존재할 수 있는 공간의 반지름

    float radius_snow;
    float radius_ice;
    float E_snow;
    float E_ice;
    float tan_ph;
    float k_q;
    float coh_snow;
    float coh_ice;
    float minForceW;
    float maxForceW;
    float heatCap_water;
    float heatCap_snow;
    float hearCap_ice;
    float tens;
    float ther_water;
    float ther_snow;
    float ther_ice;
    float threshold;
    int MaxNeighbor;

    float gravity;
    float particle_mass;
    float coefficient_friction;

    float3 startFloor;
    float3 endTop;

    int3 binSize3;
    unsigned int binLen;
    float maxSpeed = 3.f;

    unsigned int num_interacting_collider = 0;
    unsigned int num_collider = 0;
};


struct ParticleData {
    float3 positiveForce;
    float3 negativeForce;
    float phase_snow;
    int neighborCounts;
    float radius;
    float d;
    float CohNeighbor[27];
    
};

struct Collider {//파티클과 충돌하는 충돌체

    float3 vel;
    float radius;
    float3 force;//interacting collider에서만 계산
    float mass;//interacting collider에서만 계산
};

class SnowCUDA {
private:
    void calcBin(float3 startFloor, float3 endTop);

public:
    float3* dev_pos = 0;
    float3* dev_vel = 0;
    float3* dev_force = 0;

    float* dev_radius = 0;

    float3* dev_debug = 0;
    ParticleData* dev_pData = 0;
    Collider* dev_colliderNotInteracting = 0;
    Collider* dev_colliderInteracting = 0;
    float3* dev_colPos = 0;
    float3* dev_preColPos = 0; // 속도계산을 위한 이전 프레임 위치정보 저장 변수
    float3* dev_colInterPos = 0;
    ParamSet Param;


    // FRNN 관련 변수
    int* dev_PID;
    int* dev_BinID;
    int* dev_Count; // BinLen + 1 
    int* dev_PBM; // BinLen + 1 
    float* radius;

    unsigned int Max;
    unsigned int size;

    cudaError_t cuda_status;
    bool isCrashed = false;

    SnowCUDA() {}
    ~SnowCUDA() {}
    cudaError_t initSnowCUDA(unsigned int _size, unsigned int _MAX, float initPhaseSnow, float3 startFloor, float3 endTop, std::string* error_message);
    cudaError_t AddInteractingCollider(Collider* collider,float3* Colpos, std::string* error_message);
    cudaError_t AddCollider(Collider* collider, float3* Colpos, std::string* error_message);
    cudaError_t UpdateCohesionForce(float3* debug, std::string* error_message);
    cudaError_t UpdateTransform(float3* pos, float* radius,float deltaTime, std::string* error_message);
    cudaError_t UpdateColliderPosition(float3* colInterPos, float deltaTime, std::string* error_message);
    cudaError_t UpdateCollision( float3* colPos,float3* debug, float deltaTime, std::string* error_message);
    cudaError_t UpdateFriction(float3* debug, std::string* error_message);
    cudaError_t StartSimulation(float3* pos,  std::string* error_message);
    cudaError_t FRNN(std::string* error_message);
    void postErrorTask();

    cudaError_t Debug_GetParticleData(int index,ParticleData* particleData,float3* pos, float3* vel, std::string* error_message);

};