// Fill out your copyright notice in the Description page of Project Settings.


#include "BaseColliderCompo.h"

// Sets default values for this component's properties
UBaseColliderCompo::UBaseColliderCompo()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;
}


// Called when the game starts
void UBaseColliderCompo::BeginPlay()
{
	Super::BeginPlay();

	
}


// Called every frame
void UBaseColliderCompo::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

}

void UBaseColliderCompo::InitCollider(float _radius, float _mass) {
	radius = _radius;
	mass = _mass;
}


float UBaseColliderCompo::GetRadius()
{
	return radius;
}


float UBaseColliderCompo::GetMass() {
	return mass;
}

