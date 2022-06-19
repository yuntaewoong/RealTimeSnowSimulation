// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SnowActorCpp.h"
#include "SnowAreaManager.generated.h"

UCLASS()
class REALTIMESNOWUE_API ASnowAreaManager : public AActor
{
	GENERATED_BODY()
	
public:
	
	// Sets default values for this actor's properties
	ASnowAreaManager();
	~ASnowAreaManager();

	UFUNCTION(Blueprintcallable, Category = "SnowCUDA")
	void InitManager(FVector startFloor, FVector endFloor, int numHorizontalArea, int numVerticalArea,int numHorizontalHeightMap,int numVerticalHeightMap);

	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
	TArray<ASnowActorCpp*> GetSnowActors();

	UFUNCTION(Blueprintcallable,Category = "SnowCUDA")
	int GetAreaKey(FVector position);

	UFUNCTION(Blueprintcallable, Category = "SnowCUDA")
	TArray<FVector> GetVertices(int key);

	UFUNCTION(Blueprintcallable, Category = "SnowCUDA")
	TArray<int> GetIndices();

	UFUNCTION(Blueprintcallable, Category = "SnowCUDA")
	void UpdateHeightmap(TArray<float> XPos, TArray<float> YPos, TArray<float> ZPos, int key);

	UPROPERTY(BlueprintReadWrite)
	TArray<ASnowActorCpp*> m_snowActors;

	UPROPERTY(BlueprintReadOnly)
	float m_width;
	UPROPERTY(BlueprintReadOnly)
	float m_length;
	UPROPERTY(BlueprintReadOnly)
	float m_height;
	UPROPERTY(BlueprintReadOnly)
	int m_numHorizonArea;
	UPROPERTY(BlueprintReadOnly)
	int m_numVerticalArea;
	UPROPERTY(BlueprintReadOnly)
	int m_numHorizonHeightMap;
	UPROPERTY(BlueprintReadOnly)
	int m_numVerticalHeightMap;
	UPROPERTY(BlueprintReadOnly)
	float m_oneAreaWidth;
	UPROPERTY(BlueprintReadOnly)
	float m_oneAreaLength;

	UPROPERTY(BlueprintReadWrite)
	int m_initHeightZ = 135;


	int** GetHeightMap(int key);
	float GetZPositionByChar(char cha);
	int GetintByZPosition(float zPosition);
	float GetWidthPerHeightMapBlock();
	float GetLengthPerHeightMapBlock();
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	

private:
	TMap<int, int**> m_heightMaps;
	FVector m_startFloor;

	
	
};
