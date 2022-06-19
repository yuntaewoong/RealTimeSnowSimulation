// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "SnowCuda.h"
#include "GameFramework/Actor.h"
#include "cuda_runtime.h"
#include "SnowParitcleCPP.h"
#include "SnowCollider.h"
#include "ColliderInfo.h"
#include "BaseColliderCompo.h"
#include "BaseColliderActor.h"
#include "SnowActorCpp.generated.h"

#define AMOUNT 14000
UCLASS()
class REALTIMESNOWUE_API ASnowActorCpp : public AActor
{
	GENERATED_BODY()
	
public:	
	ASnowActorCpp();
	TSubclassOf<class ASnowParitcleCPP> particle;
	
	UPROPERTY(BlueprintReadWrite)
	float InitRadius;

	SnowCUDA SnowSim = SnowCUDA();
	UPROPERTY(BlueprintReadWrite)
	float init_phase = 0.8f; //입자의 눈/얼음 초기 비율
	float3 Pos[AMOUNT];
	float3 debug[AMOUNT];
	float radiusArr[AMOUNT];
	Collider ColliderInteracting[MAX_OF_COLLIDER_INTERACTING];
	float3 colPos[MAX_OF_COLLIDER_NOT_INTERACTING];
	float3 colPosInter[MAX_OF_COLLIDER_INTERACTING];
	UPROPERTY(BlueprintReadWrite)
		TArray<ASnowCollider*> ColliderArr;//눈 과의 충돌이 collider에 영향을 미치는 collider actor들

	UPROPERTY(BlueprintReadWrite)
		TArray<FColliderInfo>ColliderInfoArr;//눈과의 충돌이 collider에 영향을 미치지 않는 colldier

	UPROPERTY(BlueprintReadWrite)
		TArray<UBaseColliderCompo*> interCol;//눈 과의 충돌이 collider에 영향을 미치는 collider actor들

	UPROPERTY(BlueprintReadWrite)
		TArray<UBaseColliderCompo*> NoneCol;//눈과의 충돌이 collider에 영향을 미치지 않는 colldier

	UPROPERTY(BlueprintReadWrite)
		TArray<ABaseColliderActor*> ActorNoneCol;//눈과의 충돌이 collider에 영향을 미치지 않는 colldier ------ test

	Collider colliderInfo[MAX_OF_COLLIDER_NOT_INTERACTING];
	int numOfColliderInfo;


	UPROPERTY(BlueprintReadWrite)
	int Amount = AMOUNT;

	UPROPERTY(BlueprintReadWrite)
		TArray <FTransform> SnowTrfs;

	

	UPROPERTY(BlueprintReadWrite)
		TArray <FVector> SnowInitLoc;

	
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool InitSnow(FVector StartFloor, FVector EndTop, float friction, float Esnow, float Eice, float coh_snow, float coh_ice, float tanTheta, float RadiusSnow, float RadiusIce, float snowMass, float maxSpeed);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool AddCollider(bool Interactabe,ASnowCollider* collider);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool AddColliderCompo(bool Interactabe, UBaseColliderCompo* collider);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool AddColliderComposArr(bool Interactabe, TArray<UBaseColliderCompo*> collider);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateCohesion();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateInteractingColliderPosition(float DeltaTime);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateCollision(float dt);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateFriction();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		float GetRadius(int idx);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void SetParams(float particleMass, float friction, float Esnow, float Eice, float RadiusSnow, float RadiusIce);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void FRNN();

	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdatePositionMesh(float DeltaTime);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool StartSimulation();

	UFUNCTION(BlueprintCallable, Category = "CollisionSampling")
		void UpdateColliderInfoArr(TArray<ABaseColliderActor*> refCollider);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		TArray<float> GetXPos();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		TArray<float> GetYPos();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		TArray<float> GetZPos();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
