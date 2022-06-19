// Fill out your copyright notice in the Description page of Project Settings.


#include "SnowActorCpp.h"

// Sets default values
ASnowActorCpp::ASnowActorCpp()
	:colliderInfo{Collider(),}
	,numOfColliderInfo(0)
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void ASnowActorCpp::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ASnowActorCpp::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
void ASnowActorCpp::SetParams(float particleMass, float friction, float Esnow, float Eice, float RadiusSnow, float RadiusIce) {
	SnowSim.Param.radius_snow = RadiusSnow;
	SnowSim.Param.radius_ice = RadiusIce;
	SnowSim.Param.E_snow = Esnow;
	SnowSim.Param.E_ice = Eice;
	SnowSim.Param.particle_mass = particleMass;
	SnowSim.Param.coefficient_friction = friction;
}

UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
bool ASnowActorCpp::InitSnow(FVector StartFloor, FVector EndTop, float friction=0.5f,
	float Esnow = 5000.f, float Eice=35000.f, float coh_snow = 625.f, float coh_ice = 3750.f,float tanTheta = 38.0f,float RadiusSnow=0.05f, float RadiusIce=0.025f, float snowMass = 0.05f, float maxSpeed = 3.f ) {
	
	SnowSim.Param.radius_snow = RadiusSnow;
	SnowSim.Param.radius_ice = RadiusIce;
	SnowSim.Param.E_snow = Esnow;
	SnowSim.Param.E_ice = Eice;
	SnowSim.Param.tan_ph = tanf(tanTheta / 180.f * 3.141592);
	SnowSim.Param.k_q = 0.000005f;
	SnowSim.Param.coh_snow = coh_snow;
	SnowSim.Param.coh_ice = coh_ice;

	SnowSim.Param.heatCap_water = 4186.f;
	SnowSim.Param.heatCap_snow = 2090.f;
	SnowSim.Param.hearCap_ice = 2050.f;
	SnowSim.Param.tens = 0.00012f;
	SnowSim.Param.ther_water = 0.602f;
	SnowSim.Param.ther_snow = 0.1f;
	SnowSim.Param.ther_ice = 0.7f;
	SnowSim.Param.threshold = 0.75f;
	SnowSim.Param.MaxNeighbor = 26;
	SnowSim.Param.h = SnowSim.Param.radius_snow * 2.0;
	SnowSim.Param.maxSpeed = maxSpeed;

	SnowSim.Param.gravity = 9.8f;
	SnowSim.Param.particle_mass = snowMass;
	SnowSim.Param.coefficient_friction = friction;
	SnowSim.Param.minForceW = SnowSim.Param.particle_mass * SnowSim.Param.gravity * 1.05f;
	SnowSim.Param.maxForceW = 10000.f;
	std::string error_message;

	float3 fStartFloor = make_float3(StartFloor.X / 100.f, StartFloor.Y / 100.f, StartFloor.Z / 100.f);
	float3 fEndTop = make_float3(EndTop.X / 100.f, EndTop.Y / 100.f, EndTop.Z / 100.f);

	cudaError_t cuda_status = SnowSim.initSnowCUDA(Amount, Amount, init_phase, fStartFloor, fEndTop, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("CUDAInit failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	
	InitRadius = init_phase * SnowSim.Param.radius_snow + (1.f - init_phase) * SnowSim.Param.radius_ice;
	
	return true;
}

bool ASnowActorCpp::AddCollider(bool Interactable, ASnowCollider* collider){
	if (Interactable) {
		if (ColliderArr.Num() == MAX_OF_COLLIDER_INTERACTING)
			return false;
		ColliderArr.Add(collider);//cpu메모리에 collider정보 보관
		colPosInter[ColliderArr.Num() - 1] = make_float3(
			collider->GetActorLocation().X / 100.f,
			collider->GetActorLocation().Y / 100.f,
			collider->GetActorLocation().Z / 100.f
		);
		ColliderInteracting[ColliderArr.Num() - 1].force = make_float3(0.f, 0.f, 0.f);
		ColliderInteracting[ColliderArr.Num() - 1].mass = collider->GetMass();

		ColliderInteracting[ColliderArr.Num() - 1].radius = collider->GetRadius();
		ColliderInteracting[ColliderArr.Num() - 1].vel = make_float3(0.f, 0.f, 0.f);
		std::string error_message;
		SnowSim.AddInteractingCollider(ColliderInteracting, colPosInter, &error_message);
		SnowSim.Param.num_interacting_collider++;
		//SnowSim.Param.num_collider++;
	}

	return true;
}

bool ASnowActorCpp::AddColliderCompo(bool Interactable, UBaseColliderCompo* collider) {
	if (Interactable) {
		if (interCol.Num() == MAX_OF_COLLIDER_INTERACTING)
			return false;
		interCol.Add(collider);//cpu메모리에 collider정보 보관
		colPosInter[interCol.Num() - 1] = make_float3(
			collider->GetComponentLocation().X / 100.f,
			collider->GetComponentLocation().Y / 100.f,
			collider->GetComponentLocation().Z / 100.f
		);
		ColliderInteracting[interCol.Num() - 1].force = make_float3(0.f, 0.f, 0.f);
		ColliderInteracting[interCol.Num() - 1].mass = collider->GetMass();

		ColliderInteracting[interCol.Num() - 1].radius = collider->GetComponentScale().X/2.f;
		ColliderInteracting[interCol.Num() - 1].vel = make_float3(0.f, 0.f, 0.f);
		std::string error_message;
		SnowSim.AddInteractingCollider(ColliderInteracting, colPosInter, &error_message);
		UE_LOG(LogTemp, Warning, TEXT("%f\n"), collider->GetComponentScale().X / 2.f);
		SnowSim.Param.num_interacting_collider++;
		
	}
	else {
		if (NoneCol.Num() == MAX_OF_COLLIDER_NOT_INTERACTING)
			return false;
		NoneCol.Add(collider);//cpu메모리에 collider정보 보관
		colPos[NoneCol.Num() - 1] = make_float3(
			collider->GetComponentLocation().X / 100.f,
			collider->GetComponentLocation().Y / 100.f,
			collider->GetComponentLocation().Z / 100.f
		);
		colliderInfo[NoneCol.Num() - 1].force = make_float3(0.f, 0.f, 0.f);
		colliderInfo[NoneCol.Num() - 1].mass = collider->GetMass();

		colliderInfo[NoneCol.Num() - 1].radius = collider->GetComponentScale().X/2.f;
		colliderInfo[NoneCol.Num() - 1].vel = make_float3(0.f, 0.f, 0.f);
		std::string error_message;
		SnowSim.AddCollider(colliderInfo, colPos, &error_message);
		UE_LOG(LogTemp, Warning, TEXT("%f\n"), collider->GetComponentScale().X / 2.f);
		SnowSim.Param.num_collider++;
	}

	return true;
}

bool ASnowActorCpp::AddColliderComposArr(bool Interactable, TArray<UBaseColliderCompo*> colliders) {
	if (Interactable) {
		if ((interCol.Num() + colliders.Num()) == MAX_OF_COLLIDER_INTERACTING)
			return false;
		for (int i = 0; i < colliders.Num(); i++) {
			AddColliderCompo(Interactable, colliders[i]);
		}
	}
	else {
		if ((NoneCol.Num() + colliders.Num()) == MAX_OF_COLLIDER_NOT_INTERACTING)
			return false;
		for (int i = 0; i < colliders.Num(); i++) {
			AddColliderCompo(Interactable, colliders[i]);
		}
	}

	return true;
}


bool ASnowActorCpp::UpdateCohesion() {

	std::string error_message;
	cudaError_t cuda_status = SnowSim.UpdateCohesionForce(debug, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("CUDA Cohesion failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	return true;
}



bool ASnowActorCpp::UpdateInteractingColliderPosition(float DeltaTime) {
	std::string error_message;
	cudaError_t cuda_status = SnowSim.UpdateColliderPosition(colPosInter,DeltaTime,&error_message);

	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("Moving Update failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	for (int i = 0; i < ColliderArr.Num(); i++)
	{
		
		ColliderArr[i]->SetActorLocation(
			FVector(
				colPosInter[i].x * 100.f,
				colPosInter[i].y * 100.f,
				colPosInter[i].z * 100.f
			)
		);
		
	}
	return true;
}
bool ASnowActorCpp::UpdatePositionMesh(float DeltaTime) {
	std::string error_message;
	cudaError_t cuda_status = SnowSim.UpdateTransform(Pos, radiusArr, DeltaTime, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("Moving Mesh Update failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	for (int i = 0; i < SnowTrfs.Num(); i++)
	{
		SnowTrfs[i].SetLocation(FVector(
			Pos[i].x * 100.f,
			Pos[i].y * 100.f,
			Pos[i].z * 100.f
		));
		SnowTrfs[i].SetScale3D(FVector(
			200.f * radiusArr[i], // 기본 mesh의 길이가 256cm, 100/256 * 2 * radius
			200.f * radiusArr[i],
			200.f * radiusArr[i]
		));
	}
	return true;
}

bool ASnowActorCpp::UpdateCollision(float dt) {
	std::string error_message;
	
	for (int i = 0; i < NoneCol.Num(); i++)
	{
		colPos[i] = make_float3(NoneCol[i]->GetComponentLocation().X / 100.f, NoneCol[i]->GetComponentLocation().Y / 100.f, NoneCol[i]->GetComponentLocation().Z / 100.f);
	}
	for (int i = 0; i < ActorNoneCol.Num(); i++)
	{
		colPos[i+NoneCol.Num()] = make_float3(ActorNoneCol[i]->GetActorLocation().X / 100.f, ActorNoneCol[i]->GetActorLocation().Y / 100.f, ActorNoneCol[i]->GetActorLocation().Z / 100.f);
	}
	cudaError_t cuda_status = SnowSim.UpdateCollision(colPos,debug, dt, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("Collision Update failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	



	return true;
}
bool ASnowActorCpp::UpdateFriction() {
	std::string error_message;
	cudaError_t cuda_status = SnowSim.UpdateFriction(debug, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("Collision Update failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	return true;
}
float ASnowActorCpp::GetRadius(int idx) {
	return radiusArr[idx];
}
void ASnowActorCpp::FRNN() {
	std::string error_message;

	cudaError_t cuda_status = SnowSim.FRNN( &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("FRNN failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
	}
} 

bool ASnowActorCpp::StartSimulation() {
	std::string error_message;
	for (int i = 0; i < Amount; i++) {
		SnowTrfs.Add(FTransform());
		Pos[i].x = SnowInitLoc[i].X / 100.f;
		Pos[i].y = SnowInitLoc[i].Y / 100.f;
		Pos[i].z =  SnowInitLoc[i].Z / 100.f;
		radiusArr[i] = SnowSim.Param.radius_snow;
	}
	cudaError_t cuda_status = SnowSim.StartSimulation(Pos, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("FRNN failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
	}
	return true;
}

void ASnowActorCpp::UpdateColliderInfoArr(TArray<ABaseColliderActor*> refCollider)
{
	if (NoneCol.Num() + ActorNoneCol.Num() + refCollider.Num() >= MAX_OF_COLLIDER_NOT_INTERACTING)
	{
		UE_LOG(LogTemp, Warning, TEXT("num %d\n"), NoneCol.Num() + ActorNoneCol.Num() + refCollider.Num());
		UE_LOG(LogTemp, Warning, TEXT("UpdateColliderInfoArr failed!\n"));
		return;
	}
	for(ABaseColliderActor* refCol : refCollider)
	{
		ActorNoneCol.Add(refCol);//cpu메모리에 collider정보 보관
		colPos[ActorNoneCol.Num() - 1] = make_float3(
			refCol->GetActorLocation().X / 100.f,
			refCol->GetActorLocation().Y / 100.f,
			refCol->GetActorLocation().Z / 100.f
		);
		colliderInfo[ActorNoneCol.Num() - 1].force = make_float3(0.f, 0.f, 0.f);
		colliderInfo[ActorNoneCol.Num() - 1].mass = refCol->mass;
		colliderInfo[ActorNoneCol.Num() - 1].radius = refCol->radius / 100.f;
		colliderInfo[ActorNoneCol.Num() - 1].vel = make_float3(0.f, 0.f, 0.f);
		std::string error_message;
		SnowSim.AddCollider(colliderInfo, colPos, &error_message);
		SnowSim.Param.num_collider++;
	}
}
TArray<float> ASnowActorCpp::GetXPos()
{
	TArray<float> result;
	for (int i = 0; i < AMOUNT; i++)
		result.Add(Pos[i].x);
	return result;
}
TArray<float> ASnowActorCpp::GetYPos()
{
	TArray<float> result;
	for (int i = 0; i < AMOUNT; i++)
		result.Add(Pos[i].y);
	return result;
}
TArray<float> ASnowActorCpp::GetZPos()
{
	TArray<float> result;
	for (int i = 0; i < AMOUNT; i++)
		result.Add(Pos[i].z);
	return result;
}