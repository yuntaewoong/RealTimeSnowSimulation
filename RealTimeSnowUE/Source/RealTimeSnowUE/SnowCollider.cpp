// Fill out your copyright notice in the Description page of Project Settings.


#include "SnowCollider.h"

// Sets default values
ASnowCollider::ASnowCollider()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ASnowCollider::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ASnowCollider::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}
FVector ASnowCollider::GetColliderVelocity()
{
	return colliderVelocity;
}
float ASnowCollider::GetRadius()
{
	return radius;
}

void ASnowCollider::SetVelocity(FVector velocity) {
	colliderVelocity = velocity;
}
float ASnowCollider::GetMass() {
	return mass;
}