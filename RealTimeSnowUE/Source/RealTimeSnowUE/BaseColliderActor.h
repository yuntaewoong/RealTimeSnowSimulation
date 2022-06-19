// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BaseColliderActor.generated.h"

UCLASS()
class REALTIMESNOWUE_API ABaseColliderActor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ABaseColliderActor();

	UPROPERTY(BlueprintReadWrite)
		float radius = 0.5f;

	UPROPERTY(BlueprintReadWrite)
		float mass = 0.575f;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
