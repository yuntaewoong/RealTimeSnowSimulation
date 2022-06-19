// Fill out your copyright notice in the Description page of Project Settings.

#include "BaseColliderActor.h"
#include "Components/SphereComponent.h"

// Sets default values
ABaseColliderActor::ABaseColliderActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	USphereComponent* SphereComponent = CreateDefaultSubobject<USphereComponent>(TEXT("Root"));
	RootComponent = SphereComponent;
}

// Called when the game starts or when spawned
void ABaseColliderActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABaseColliderActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

