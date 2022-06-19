// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "ColliderInfo.generated.h"

USTRUCT(BlueprintType)
struct FColliderInfo
{
public:
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ColliderInfo")
		float radius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ColliderInfo")
		FVector position;
};

