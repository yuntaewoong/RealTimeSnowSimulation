// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "SnowActorCpp.h"
#include "Components/ActorComponent.h"
#include "ColliderInfo.h"
#include "CollisionSamplingManager.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class REALTIMESNOWUE_API UCollisionSamplingManager : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UCollisionSamplingManager();
	
	UPROPERTY(BlueprintReadWrite)
	TArray<FColliderInfo> ColliderArr;

	UPROPERTY(BlueprintReadWrite)
	float defaultRadius = 1.0f;	//Box¿¡ »ç¿ëµÊ

	UPROPERTY(BlueprintReadWrite)
	int nSpherePerLine = 12;	//capsule¿¡ »ç¿ëµÊ

	UFUNCTION(BlueprintCallable, Category = "CollisionSampling")
	void CollisionSampling(TArray<UPrimitiveComponent*> collisionComponents);

	UFUNCTION(BlueprintCallable, Category = "CollisionSampling")
	void Debug(float deltaTime = -1);

protected:
	void BoxCollisionSampling(UPrimitiveComponent* collisionShape);
	void SphereCollisionSampling(UPrimitiveComponent* collisionShape);
	void CapsuleCollisionSampling(UPrimitiveComponent* collisionShape);
	void CapsuleCollisionSampling2(UPrimitiveComponent* collisionShape);
public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

		
};
