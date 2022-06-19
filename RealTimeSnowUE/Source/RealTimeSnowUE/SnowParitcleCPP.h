// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SnowCuda.h"
#include "SnowParitcleCPP.generated.h"

UCLASS()
class REALTIMESNOWUE_API ASnowParitcleCPP : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ASnowParitcleCPP();

	UPROPERTY(BlueprintReadWrite)
		int index;

private:
	
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
