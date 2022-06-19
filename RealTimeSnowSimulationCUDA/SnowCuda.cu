// Copyright SCIEMENT, Inc.
// by Hirofumi Seo, M.D., CEO & President

#include "Header/SnowCuda.h"
#include "header/Common.cuh"
#include "header/CountNeighbor.cuh"
#include "header/CohesionForce.cuh"
#include "header/InitParticle.cuh"
#include "header/Compression.cuh"
#include "header/MoveParticle.cuh"
#include "header/Collision.cuh"
#include "header/friction.cuh"
#include "header/FRNN.cuh"
#include "header/MoveInteractingCollider.cuh"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>




cudaError_t SnowCUDA::initSnowCUDA(unsigned int _size, unsigned int _Max, float initPhaseSnow,float3 startFloor , float3 endTop ,std::string* error_message) {
    Max = _Max;
    size = _size;

    calcBin(startFloor, endTop);

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_pos, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc pos failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_vel, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc vel failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_force, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc vel failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_radius, Max * sizeof(float));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc vel failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_pData, Max * sizeof(ParticleData));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc neighborCounts failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_debug, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc debug failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_colliderInteracting, MAX_OF_COLLIDER_INTERACTING * sizeof(Collider));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc Collider failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMalloc((void**)&dev_colliderNotInteracting, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(Collider));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc Collider failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMalloc((void**)&dev_colInterPos, MAX_OF_COLLIDER_INTERACTING * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc Collider failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMalloc((void**)&dev_colPos, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc Collider failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMalloc((void**)&dev_preColPos, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc Collider failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMalloc((void**)&dev_PID, Max * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc PID failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_BinID, Max * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc BinID failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_Count, (Param.binLen + 1) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc binLen failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_PBM, (Param.binLen + 1) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc PBM failed!";
        postErrorTask();
        return cuda_status;
    }

    initParticles << <GRID_DIM, BLOCK_DIM >> > (dev_pData,dev_vel, dev_radius, initPhaseSnow, size, Param);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel initParticles launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "initParticles cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


}

cudaError_t SnowCUDA::AddInteractingCollider(Collider* collider, float3* Colpos, std::string* error_message) {
    cuda_status = cudaMemcpy(dev_colliderInteracting, collider, MAX_OF_COLLIDER_INTERACTING * sizeof(Collider), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dev_colInterPos, Colpos, MAX_OF_COLLIDER_INTERACTING * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    return cuda_status;
}

cudaError_t SnowCUDA::AddCollider(Collider* collider, float3* Colpos, std::string* error_message) {
    cuda_status = cudaMemcpy(dev_colliderNotInteracting, collider, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(Collider), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dev_colPos, Colpos, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(float3), cudaMemcpyHostToDevice);
    cuda_status = cudaMemcpy(dev_preColPos, Colpos, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    return cuda_status;
}


cudaError_t SnowCUDA::StartSimulation(float3* pos, std::string* error_message) {
    cuda_status = cudaMemcpy(dev_pos, pos, Max * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    initNeighbors << <GRID_DIM, BLOCK_DIM >> > (dev_pos, dev_pData, size);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel initNeighbors launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "initNeighbors cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }

}


cudaError_t SnowCUDA::UpdateCohesionForce(float3* debug, std::string* error_message) {

    if (isCrashed) return cuda_status;


    countNeighbors << <GRID_DIM, BLOCK_DIM >> > (dev_pos, dev_pData, dev_PID, dev_BinID, dev_PBM, size, Param);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel initParticles launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "initParticles cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


    getCohesionForce2 << <GRID_DIM, BLOCK_DIM >> > (dev_pos,dev_vel,dev_force, dev_pData, dev_PID,dev_BinID, dev_PBM ,dev_debug, size, Param);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }



    cuda_status = cudaMemcpy(debug, dev_debug, Max * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy debug D2H failed!";
        postErrorTask();
        return cuda_status;
    }

    return cuda_status;
}


cudaError_t SnowCUDA::UpdateTransform(float3* pos, float* radius, float deltaTime,  std::string* error_message)
{
    if (isCrashed) return cuda_status;
    moveParticles << <GRID_DIM, BLOCK_DIM >> > (dev_pos,dev_vel, dev_force, dev_pData, deltaTime,dev_debug,size,Param);//커널 실행
    cuda_status = cudaDeviceSynchronize();//디바이스 실행완료 기다리기

    compression << <GRID_DIM, BLOCK_DIM >> > (dev_pData, dev_radius, dev_debug, size, Param);//커널 실행(dev_pData는 디바이스에 있던 데이터 그대로 사용)
    cuda_status = cudaDeviceSynchronize();//디바이스 실행완료 기다리기


    cuda_status = cudaMemcpy(pos, dev_pos, Max * sizeof(float3), cudaMemcpyDeviceToHost);//실행결과를 host로 출력
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy pos D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMemcpy(radius, dev_radius, Max * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy radius D2H failed!";
        postErrorTask();
        return cuda_status;
    }

    return cuda_status;
}

cudaError_t SnowCUDA::UpdateColliderPosition(float3* colInterPos, float deltaTime, std::string* error_message) {
    moveInteractingColliders << <1, MAX_OF_COLLIDER_INTERACTING >> > (dev_colliderInteracting, dev_colInterPos,deltaTime, dev_debug, Param);
    cuda_status = cudaDeviceSynchronize();//디바이스 실행완료 기다리기
    cuda_status = cudaMemcpy(colInterPos, dev_colInterPos, MAX_OF_COLLIDER_INTERACTING * sizeof(float3), cudaMemcpyDeviceToHost);//collider들 위치 정보 cpu로 전송

    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy radius D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    return cuda_status;
}

cudaError_t SnowCUDA::UpdateCollision(float3* colPos, float3* debug,float dt, std::string* error_message) {
    if (isCrashed) return cuda_status;
    
    cuda_status = cudaMemcpy(dev_colPos, colPos, MAX_OF_COLLIDER_NOT_INTERACTING * sizeof(float3), cudaMemcpyHostToDevice);//device의 dev_collider업데이트
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }


    calcColliderVelocity << <1, BLOCK_DIM >> > (dev_colPos, dev_preColPos, dev_colliderNotInteracting, Param, dt);
    cuda_status = cudaDeviceSynchronize();



    //dev_colliderInteracting은 gpu메모리의 데이터 업데이트 없이 그대로 사용
    collide << <GRID_DIM, BLOCK_DIM >> > (
        dev_pos,
        dev_vel,
        dev_force,
        dev_colliderNotInteracting,
        dev_colliderInteracting,
        dev_colPos,
        dev_colInterPos,
        dev_pData,
        size, 
        Param
    );//충돌!
    cuda_status = cudaDeviceSynchronize();//디바이스 실행완료 기다리기

    return cuda_status;
}


cudaError_t SnowCUDA::UpdateFriction(float3* debug, std::string* error_message) {
    if (isCrashed) return cuda_status;
    friction << <GRID_DIM, BLOCK_DIM >> > (dev_pos, dev_vel, dev_force, dev_pData, dev_debug, size, Param);//커널 실행
    cuda_status = cudaDeviceSynchronize();//디바이스 실행완료 기다리기
    cuda_status = cudaMemcpy(debug, dev_debug, Max * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }

    return cuda_status;
}


void SnowCUDA::postErrorTask() {
    cudaFree(dev_pos);
    cudaFree(dev_vel);
    cudaFree(dev_pData);
    cudaFree(dev_debug);
    cudaFree(dev_colliderNotInteracting);
    cudaFree(dev_colliderInteracting);
    cudaFree(dev_PID);
    cudaFree(dev_BinID);
    cudaFree(dev_Count);
    cudaFree(dev_PBM);

    isCrashed = true;
}

cudaError_t SnowCUDA::Debug_GetParticleData(int index, ParticleData* particleData, float3* pos, float3* vel,  std::string* error_message) {
    if (isCrashed) return cuda_status;
    cuda_status = cudaMemcpy(particleData, &dev_pData[index], sizeof(ParticleData), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMemcpy(pos, &dev_pos[index], sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMemcpy(vel, &dev_vel[index], sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
    return cuda_status;
}


cudaError_t SnowCUDA::FRNN(std::string* error_message) {

    if (isCrashed) return cuda_status;

    resetPBM << <GRID_DIM, BLOCK_DIM >> > (dev_PBM, dev_Count, Param.binLen);

    getBinLoc << <GRID_DIM, BLOCK_DIM >> > (dev_pos, dev_BinID, dev_PID, size, Param);

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }

    thrust::device_ptr<int> dev_BinID_ptr(dev_BinID);
    thrust::device_ptr<int> dev_PID_ptr(dev_PID);

    thrust::sort_by_key(dev_BinID_ptr, dev_BinID_ptr + size, dev_PID_ptr);
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }

    countElem << <GRID_DIM, BLOCK_DIM >> > (dev_BinID, dev_Count, size);


    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, dev_Count, dev_PBM, Param.binLen + 1);
    // Scan 계산을 위한 메모리를 확인 

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // 메모리 할당

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, dev_Count, dev_PBM, Param.binLen + 1);
    // Scan 계산

    cudaFree(d_temp_storage);
    // Scan 계산용 임시 메모리 해제



    return cuda_status;

}

