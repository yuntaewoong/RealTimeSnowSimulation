// Fill out your copyright notice in the Description page of Project Settings.

#include "CollisionSamplingManager.h"
#include "DrawDebugHelpers.h"

UCollisionSamplingManager::UCollisionSamplingManager()
{
	PrimaryComponentTick.bCanEverTick = true;
}

// Called every frame
void UCollisionSamplingManager::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

void UCollisionSamplingManager::CollisionSampling(TArray<UPrimitiveComponent*> collisionComponents)
{
	FCollisionShape collisionShape;
	ColliderArr.Empty();
	for (UPrimitiveComponent* collider : collisionComponents)
	{
		collisionShape = collider->GetCollisionShape();
		if (collisionShape.IsBox())
		{
			//UE_LOG(LogTemp, Log, TEXT("Box"));
			BoxCollisionSampling(collider);
		}
		else if (collisionShape.IsSphere())
		{
			//UE_LOG(LogTemp, Log, TEXT("Sphere"));
			SphereCollisionSampling(collider);
		}
		else if (collisionShape.IsCapsule())
		{
			//UE_LOG(LogTemp, Log, TEXT("Capsule"));
			CapsuleCollisionSampling(collider);
			//CapsuleCollisionSampling2(collider);
		}
	}
}

void UCollisionSamplingManager::BoxCollisionSampling(UPrimitiveComponent* collider)
{
	FVector BoxSize = collider->GetCollisionShape().GetBox() * 2;
	FVector ComponentScale = collider->GetComponentScale();

	FVector axis_z = FVector::ZeroVector;
	FVector axis_y = FVector::ZeroVector;
	FVector axis_x = FVector::ZeroVector;

	//BoxSize = FVector(BoxSize.X * ComponentScale.X, BoxSize.Y * ComponentScale.Y, BoxSize.Z * ComponentScale.Z);

	// find min
	float min = BoxSize.X;
	if (min > BoxSize.Y)min = BoxSize.Y;
	if (min > BoxSize.Z)min = BoxSize.Z;

	// set radius of sphere
	FColliderInfo info;
	if (defaultRadius > min)info.radius = min;
	else info.radius = defaultRadius;

	// set distance
	float dist = info.radius * 3.0f / 2.0f; // sphere 사이의 거리

	// set xCount, yCount, zCount
	int xCount = floor(BoxSize.X / dist);
	int yCount = floor(BoxSize.Y / dist);
	int zCount = floor(BoxSize.Z / dist);
	if (xCount * dist < BoxSize.X)xCount++;
	if (yCount * dist < BoxSize.Y)yCount++;
	if (zCount * dist < BoxSize.Z)zCount++;

	// set position of sphere
	FVector startPos = collider->GetComponentLocation() - BoxSize/2.0f; 
	FVector CenterPos = collider->GetComponentLocation();
	FRotator CapsuleRotation = collider->GetComponentRotation(); // rotation stored in degrees

	float x = 0.0f, y = 0.0f, z = 0.0f;
	for (int zIdx = 0; zIdx < zCount; zIdx++)
	{
		for (int xIdx = 0; xIdx < xCount; xIdx++)
		{
			for (int yIdx = 0; yIdx < yCount; yIdx++)
			{
				if ((zIdx != 0 && zIdx != zCount - 1) && (xIdx != 0 && xIdx != xCount - 1) && (yIdx != 0 && yIdx != yCount - 1))
				{
					continue;
				}

				// position 결정 
				x = info.radius + dist * xIdx;
				y = info.radius + dist * yIdx;
				z = info.radius + dist * zIdx;
				if (x + info.radius > BoxSize.X)
				{
					x = BoxSize.X - info.radius;
				}
				if (y + info.radius > BoxSize.Y)
				{
					y = BoxSize.Y - info.radius;
				}
				if (z + info.radius > BoxSize.Z)
				{
					z = BoxSize.Z - info.radius;
				}
				info.position = startPos + FVector(x, y, z);

				// rotation 
				info.position -= CenterPos;
				// z축 회전
				axis_z = FVector(0, 0, 1);
				info.position = info.position.RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
				// y축 회전
				axis_y = FVector(0,1,0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
				info.position = info.position.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
				// x축 회전
				axis_x = FVector(1, 0, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
				axis_x = axis_x.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
				info.position = info.position.RotateAngleAxis(-CapsuleRotation.Roll, axis_x);
				info.position += CenterPos;

				// 배열에 추가
				ColliderArr.Emplace(info);
			}
		}
	}

}

void UCollisionSamplingManager::SphereCollisionSampling(UPrimitiveComponent* collider)
{
	FColliderInfo info;
	info.radius = collider->GetCollisionShape().GetSphereRadius();
	info.position = collider->GetComponentLocation();
	ColliderArr.Emplace(info);
}

void UCollisionSamplingManager::CapsuleCollisionSampling(UPrimitiveComponent* collider)
{
	float radius = collider->GetCollisionShape().GetCapsuleRadius();
	float halfHeight = collider->GetCollisionShape().GetCapsuleHalfHeight();
	FVector CapsuleLocation = collider->GetComponentLocation();
	FRotator CapsuleRotation = collider->GetComponentRotation(); // rotation stored in degrees

	FVector axis_z = FVector::ZeroVector;
	FVector axis_y = FVector::ZeroVector;
	FVector axis_x = FVector::ZeroVector;

	FColliderInfo info;
	info.radius = radius;

	FVector refPosition = CapsuleLocation + FVector(0, 0, -halfHeight + radius);
	bool IsDone = false;
	int multi = 0;

	while (!IsDone)
	{
		// 위치 지정
		if ((-halfHeight + radius + multi * radius) > halfHeight - radius)
		{
			info.position = CapsuleLocation + FVector(0, 0, halfHeight - radius);
			IsDone = true;
		}
		else
		{
			info.position = refPosition + FVector(0, 0, multi * radius);
		}
		// rotation 
		info.position -= CapsuleLocation;
		// z축 회전
		axis_z = FVector(0, 0, 1);
		info.position = info.position.RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
		// y축 회전
		axis_y = FVector(0, 1, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
		info.position = info.position.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
		// x축 회전
		axis_x = FVector(1, 0, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
		axis_x = axis_x.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
		info.position = info.position.RotateAngleAxis(-CapsuleRotation.Roll, axis_x);
		info.position += CapsuleLocation;
		
		ColliderArr.Emplace(info);
		multi += 1;
	}
}

void UCollisionSamplingManager::CapsuleCollisionSampling2(UPrimitiveComponent* collider)
{
	float radius = collider->GetCollisionShape().GetCapsuleRadius();
	float halfHeight = collider->GetCollisionShape().GetCapsuleHalfHeight();
	FVector CapsuleLocation = collider->GetComponentLocation();
	FRotator CapsuleRotation = collider->GetComponentRotation(); // rotation stored in degrees

	FVector axis_z = FVector::ZeroVector;
	FVector axis_y = FVector::ZeroVector;
	FVector axis_x = FVector::ZeroVector;

	//위 아래 큰 Sphere 추가
	FColliderInfo info;
	info.radius = radius;

	for (UINT zIdx = 0; zIdx < 2; ++zIdx)
	{
		if (zIdx == 0) info.position = CapsuleLocation + FVector(0, 0, -halfHeight + radius);
		else info.position = CapsuleLocation + FVector(0, 0, halfHeight - radius);
		// rotation 
		info.position -= CapsuleLocation;
		// z축 회전
		axis_z = FVector(0, 0, 1);
		info.position = info.position.RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
		// y축 회전
		axis_y = FVector(0, 1, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
		info.position = info.position.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
		// x축 회전
		axis_x = FVector(1, 0, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
		axis_x = axis_x.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
		info.position = info.position.RotateAngleAxis(-CapsuleRotation.Roll, axis_x);
		info.position += CapsuleLocation;
		ColliderArr.Emplace(info);
	}
	
	//중간 원기둥 채우기
	//반지름 결정하기
	info.radius = 3.14f * radius / (3.14f + (float)nSpherePerLine);

	//시작 위치 지정
	FVector refPosition = CapsuleLocation + FVector(0, 0, -halfHeight + radius);

	//z축 방향의 개수 정하기
	float dist = info.radius * 2;
	float height = (halfHeight - radius) * 2.0f;
	int zCount = floor(height / dist);
	if (zCount * dist < height)zCount++;

	//원기둥 채우기
	float degree = 0.0f;
	for (int zIdx = 0; zIdx < zCount; ++zIdx)
	{
		for (int idx = 0; idx < nSpherePerLine; ++idx)
		{
			// 위치 지정
			degree = 360.0f / (float)nSpherePerLine * (float)idx;
			if (info.radius + zIdx * dist > height)
			{
				info.position = refPosition + FVector(0, radius - info.radius, height - info.radius) - CapsuleLocation;
			}
			else
			{
				info.position = refPosition + FVector(0, radius - info.radius, info.radius + zIdx * dist) - CapsuleLocation;
			}
			info.position = info.position.RotateAngleAxis(degree, FVector(0, 0, 1));
			
			// 회전
			// z축 회전
			axis_z = FVector(0, 0, 1);
			info.position = info.position.RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
			// y축 회전
			axis_y = FVector(0, 1, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
			info.position = info.position.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
			// x축 회전
			axis_x = FVector(1, 0, 0).RotateAngleAxis(CapsuleRotation.Yaw, axis_z);
			axis_x = axis_x.RotateAngleAxis(-CapsuleRotation.Pitch, axis_y);
			info.position = info.position.RotateAngleAxis(-CapsuleRotation.Roll, axis_x);
			info.position += CapsuleLocation;
			ColliderArr.Emplace(info);
		}
	}
}

void UCollisionSamplingManager::Debug(float deltaTime)
{
	// deltaTime -1 == 유지
	for (FColliderInfo collider : ColliderArr)
	{
		UWorld* world = GetWorld();
		if (world)
		{
			DrawDebugSphere(world, collider.position, collider.radius, 26, FColor(181, 0, 0), true, deltaTime,0, 1);
		}
	}
}