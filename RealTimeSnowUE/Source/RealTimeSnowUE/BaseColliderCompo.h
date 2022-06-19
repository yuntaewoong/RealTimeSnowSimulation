// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "BaseColliderCompo.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class REALTIMESNOWUE_API UBaseColliderCompo : public USceneComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UBaseColliderCompo();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void InitCollider(float _radius, float _mass);



	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		float GetRadius();





	float GetMass();
protected:
	// Called when the game starts
	virtual void BeginPlay() override;
private:
	FVector beforePos = FVector(0.0f, 0.0f, 0.0f);
	float radius = 0.05f;
	float mass = 0.575f;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

		
};
