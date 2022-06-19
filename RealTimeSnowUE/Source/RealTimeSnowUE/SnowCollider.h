// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SnowCollider.generated.h"

UCLASS()
class REALTIMESNOWUE_API ASnowCollider : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ASnowCollider();

	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
	FVector GetColliderVelocity();
	
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
	float GetRadius();
	
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
	void SetVelocity(FVector velocity);

	float GetMass();


private:
	
	FVector colliderVelocity = FVector(0.0f, 0.0f, 0.0f);
	float radius = 0.5f;
	float mass = 0.575f;
	
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
