/*
IndicesArray is an array that contains all indices of triangles combined from all individual arrays in all octants, the values here point to the triangles array
every octant has an extra variable "indicesStart" which points to the location in IndicesArray from which the triangles belonging to the octant start, and also 
there is a length variable to know where the list ends
so on the GPU we have 3 big arrays, an array of Octants,an array of Triangles and an array of indices. The octants fetch an index, then fetch the triangle corresponding to it, then test
according to the test result, the value in the indicesArray is either left intact or changed to -1. This array is then copied back to the CPU
The -1s are the locations of the indices that should be removed from the original individual arrays of each octant, that is done in C#

*/



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define NUMOFOCTANTS 2000000 
#define THREADSPERBLOCK 512


 struct GPUinterval
 {
	__device__ GPUinterval()
		{
		}
		float StartValue;
        float EndValue;
 };

  struct GPUrectBoundingBox
 {
	__device__ GPUrectBoundingBox()
		{
		}
		float3 maxpoint;
        GPUinterval intervalX,intervalY,intervalZ;
 };

struct GPUTriangle
{
	__device__ GPUTriangle()
	{
	}
	float3 A;
	float3 B;
	float3 C;
	float3 normal;
	GPUrectBoundingBox rectBoundingBox;
};

struct GPUCubicBoundingBox
{
	__device__ GPUCubicBoundingBox()
	{
	}
	float Length;
	float3 MaxPoint;
	GPUinterval intervalX,intervalY,intervalZ;
};

struct GPUOctant
{
	__device__ GPUOctant()
	{
	}
	GPUCubicBoundingBox CubicBoundingBox;
	int indicesStart;
	int LengthOfIndicesArray;
};

extern "C" __declspec(dllexport) __global__ void SAT(GPUOctant* gOctants, GPUTriangle* gTriangles,int* gIndices,int gOctantsLength,int gTrianglesLength,int gIndicesLength);

inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ void CrossMultiply(float* a, float3* b,float3* result)
        {

			result->x=a[1]*b->z-a[2]*b->x;
			result->y=a[2]*b->x-a[0]*b->z;
			result->z=a[0]*b->y-a[1]*b->x;
			
        }

    __device__ void minMax(float* min,float* max, float* value1, float* value2, float* value3)
         {
             int v1 = (int)(*value1 * 100000);
             int v2 = (int)(*value2 * 100000);
             int v3 = (int)(*value3 * 100000);
             
             *max = v1 - ((v1 - v2) & ((v1 - v2) >> (sizeof(int) * 8 - 1)));
             *min = v2 + ((v1 - v2) & ((v1 - v2) >> (sizeof(int) * 8 - 1)));

             *max = *max - (((int)*max - v3) & (((int)*max - v3) >> (sizeof(int) * 8 - 1)));
             *min = v3 + (((int)*min - v3) & (((int)*min - v3) >> (sizeof(int) * 8 - 1)));

             *max = *max / 100000;
             *min = *min / 100000;
         }

__device__ bool intervalIntersects(float3* a0A, float* v0, float* v1, float* v2, float* halfDiagonal)
         {
			 float ps0 = a0A->x * v0[0] + a0A->y * v0[1] + a0A->z * v0[2];
             float ps1 = a0A->x * v1[0] + a0A->y * v1[1] + a0A->z * v1[2];
             float ps2 = a0A->x * v2[0] + a0A->y * v2[1] + a0A->z * v2[2];

             //var pMin = minimum(ps0, ps1, ps2);
             //var pMax = maximum(ps0, ps1, ps2);
              float pMin=0, pMax=0;
             //a little loss of precision happens here
             minMax(&pMin, &pMax, &ps0, &ps1, &ps2);
             //need to see how to use the cuda equivalent of absolute
             float r = *halfDiagonal * fabsf(a0A->x) + *halfDiagonal * fabsf(a0A->y) + *halfDiagonal * fabsf(a0A->z);

             return pMin > r || pMax < -r;
         }


__device__ bool ValueIntersectsInterval(float value, GPUinterval interval)
{
	return value >= interval.StartValue && value <= interval.EndValue;
}

__device__ bool IntervalIntersectsInterval(GPUinterval intervalA, GPUinterval intervalB)
{
	
            if (ValueIntersectsInterval(intervalA.StartValue, intervalB))
                return true;

            if (ValueIntersectsInterval(intervalA.EndValue, intervalB))
                return true;

            return intervalA.StartValue < intervalB.StartValue && intervalA.EndValue > intervalB.EndValue;
}


__device__ bool BoxIntersectsBox(GPUrectBoundingBox rectangularBoundingBoxA, GPUCubicBoundingBox cubicBoundingBox)
        {
            return IntervalIntersectsInterval(rectangularBoundingBoxA.intervalX, cubicBoundingBox.intervalX) &&
                   IntervalIntersectsInterval(rectangularBoundingBoxA.intervalY, cubicBoundingBox.intervalY) &&
                   IntervalIntersectsInterval(rectangularBoundingBoxA.intervalZ, cubicBoundingBox.intervalZ);
        }

__device__ bool PlaneIsIntersecting(GPUCubicBoundingBox cubicBoundingBox, GPUTriangle* triangle,float3 center,float halfDiagonal)
{
	float planD=dot(triangle->normal,triangle->A);
	float e=halfDiagonal*fabsf(triangle->normal.x)+halfDiagonal*fabsf(triangle->normal.y)+halfDiagonal*fabsf(triangle->normal.z);
	float s=dot(center,triangle->normal)-planD;
	 if (s - e > 0)
                return false;
     if (s + e < 0)
                return false;

            return true;
}

__device__ bool IsIntersecting(GPUOctant* goctant,GPUTriangle* gtriangle)
{
			float3* maxPoint = &(goctant->CubicBoundingBox.MaxPoint);
            float halfSideLength = (goctant->CubicBoundingBox.Length) / 2;
            float center[3]={maxPoint->x - halfSideLength,maxPoint->y - halfSideLength,maxPoint->z - halfSideLength};
            float halfDiagonal = goctant->CubicBoundingBox.Length / 2;
			float v0[3] = {gtriangle->A.x-center[0],gtriangle->A.y-center[1],gtriangle->A.z-center[2]};
			float v1[3]= {gtriangle->B.x-center[0],gtriangle->B.y-center[1],gtriangle->B.z-center[2]};
			float v2[3]= {gtriangle->C.x-center[0],gtriangle->C.y-center[1],gtriangle->C.z-center[2]};
            float e0 [3]=  { 1, 0, 0 };
            float e1 [3]=  { 0, 1, 0 };
            float e2 [3] = { 0, 0, 1 };
			float3 side,center2;
			side.x=gtriangle->B.x-gtriangle->A.x;
			side.y=gtriangle->B.y-gtriangle->A.y;
			side.z=gtriangle->B.z-gtriangle->A.z;
			//BoxIntersectsBox test
			 if (!BoxIntersectsBox(gtriangle->rectBoundingBox, goctant->CubicBoundingBox))
                return false;
			 //Plane intersects Box test
			 center2.x=center[0];center2.y=center[1];center2.z=center[2];
			 if(!PlaneIsIntersecting(goctant->CubicBoundingBox,gtriangle,center2,halfDiagonal))
				 return false;


			 //triangle bounding box intersection test
			//AB 
			float3 a0A;
			CrossMultiply(e0, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;

            CrossMultiply(e1, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;

			
            CrossMultiply(e2, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;
			//BC
			side.x=gtriangle->B.x-gtriangle->C.x;
			side.y=gtriangle->B.y-gtriangle->C.y;
			side.z=gtriangle->B.z-gtriangle->C.z;

			CrossMultiply(e0, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;

            //var a0C = e0.CrossMultiply(f2);
            CrossMultiply(e0, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;

          

            //var a1B = e1.CrossMultiply(f1);
            CrossMultiply(e1, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;

            //AC
			side.x=gtriangle->C.x-gtriangle->A.x;
			side.y=gtriangle->C.y-gtriangle->A.y;
			side.z=gtriangle->C.z-gtriangle->A.z;

            CrossMultiply(e1, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;


            //var a2B = e2.CrossMultiply(f1);
            CrossMultiply(e2, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;

            //var a2C = e2.CrossMultiply(f2);
            CrossMultiply(e2, &side,&a0A);
            if (intervalIntersects(&a0A, v0, v1, v2, &halfDiagonal))
                return false;


			return true;
}


extern "C" __global__ void SAT(GPUOctant* gOctants, GPUTriangle* gTriangles,int* gIndices,int gOctantsLength,int gTrianglesLength,int gIndicesLength)
{
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	GPUOctant goctant;
	int triangleIndex;
	if (threadID < gOctantsLength)
	{
		goctant = gOctants[(threadID)]; //main memory access 
		int i = 0;
		for (i = 0; i < goctant.LengthOfIndicesArray; i++)
		{
			triangleIndex = gIndices[goctant.indicesStart+i]; //main memory access
			GPUTriangle gtriangle = gTriangles[triangleIndex]; //main memory access
			if (!IsIntersecting(&goctant, &gtriangle))
			{
				__syncthreads();
				gIndices[goctant.indicesStart+i] = -1; //main memory write
			}

		}
	}
}


 extern"C" __declspec(dllexport) void __stdcall
 runGPUIntersectionTest(GPUOctant octants[],GPUTriangle triangles[],int indices[],int lengthOfOctants,int lengthOfTriangles,int lengthOfIndices)
{
	cudaError_t cudaStatus;
	//Memory allocation and copying------------------------------------------------------------
	GPUOctant* gOctants;
	GPUTriangle* gTriangles;
	int* gIndices;
	cudaStatus=cudaMalloc((void**)&gOctants,sizeof(GPUOctant)*lengthOfOctants);
	cudaStatus=cudaMalloc((void**)&gTriangles,sizeof(GPUTriangle)*lengthOfTriangles);
	cudaStatus=cudaMalloc((void**)&gIndices,sizeof(int)*lengthOfIndices);
	if (cudaStatus == cudaSuccess) {
        printf("allocation success!\n");
    }
	else {printf("allocation failed\n");}


	cudaStatus=cudaMemcpy(gOctants,octants,sizeof(GPUOctant)*lengthOfOctants,cudaMemcpyHostToDevice);
	cudaStatus=cudaMemcpy(gTriangles,triangles,sizeof(GPUTriangle)*lengthOfTriangles,cudaMemcpyHostToDevice);
	cudaStatus=cudaMemcpy(gIndices,indices,sizeof(int)*lengthOfIndices,cudaMemcpyHostToDevice);
	if (cudaStatus == cudaSuccess) {
        printf("memory copy success!\n");
    }
	else {printf("memory copy failed\n");}

	//---------------------------------------------------------------------------------------------
	//kernel launching
	dim3 gridDim, blockDim;
	int n = lengthOfOctants;
            if(n>512)  
            {
                blockDim=dim3(THREADSPERBLOCK);
                int numberOfBlocks = n / THREADSPERBLOCK;
                int remainder = n % THREADSPERBLOCK;
                if (remainder == 0)
                    gridDim =  dim3(numberOfBlocks);
                else gridDim =  dim3(numberOfBlocks + 1);
            }
            else 
            {
                blockDim =  dim3(n);
                gridDim =  dim3(1);
            }


			SAT<<<gridDim,blockDim>>>(gOctants,gTriangles,gIndices,lengthOfOctants,lengthOfTriangles,lengthOfIndices);

			cudaStatus=cudaDeviceSynchronize();
			 if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
			 else{printf("launch terminated successfully\n");}
			//----------------------------------------------------------------------------------
			//copy back and free gpu memory
			 cudaMemcpy(indices,gIndices,sizeof(int)*lengthOfIndices,cudaMemcpyDeviceToHost);
			 cudaFree(gOctants);
			 cudaDeviceReset();
			//at this point, the list of indices should contain some -1 values, these do not intersect
			 // this change is reflected in the C# code, for refinment
		
}




int main()
{
	//for testing purposes
   	GPUOctant* goct=(GPUOctant*)calloc(NUMOFOCTANTS,sizeof(GPUOctant));
	for(int i=0;i<NUMOFOCTANTS;i++)
	{
		goct[i].CubicBoundingBox.MaxPoint.x=1.2;
		goct[i].CubicBoundingBox.MaxPoint.y=1.1;
		goct[i].CubicBoundingBox.MaxPoint.z=1.0;
		goct[i].CubicBoundingBox.Length=5.2;
		goct[i].CubicBoundingBox.intervalX.StartValue=1.1;
		goct[i].CubicBoundingBox.intervalX.EndValue=42.1;
		goct[i].CubicBoundingBox.intervalY.StartValue=1.1;
		goct[i].CubicBoundingBox.intervalY.EndValue=42.1;
		goct[i].CubicBoundingBox.intervalZ.StartValue=1.1;
		goct[i].CubicBoundingBox.intervalZ.EndValue=42.1;
		goct[i].LengthOfIndicesArray=62;
		goct[i].indicesStart=0;
	}

	int* gindices=(int*)calloc(62,sizeof(int));
	GPUTriangle* gtri=(GPUTriangle*)calloc(62,sizeof(GPUTriangle));
	for(int i=0;i<62;i++)
	{
		gindices[i]=i;
		gtri[i].A.x=2.2;
		gtri[i].A.y=1.2;
		gtri[i].A.z=0.2;
		gtri[i].B.x=2.2;
		gtri[i].B.y=1.2;
		gtri[i].B.z=0.2;
		gtri[i].C.x=2.2;
		gtri[i].C.y=1.2;
		gtri[i].C.z=0.2;
		gtri[i].normal.x=5.2;
		gtri[i].normal.y=1.2;
		gtri[i].normal.z=9.2;
		gtri[i].rectBoundingBox.maxpoint.x=1.2;
		gtri[i].rectBoundingBox.maxpoint.y=1.1;
		gtri[i].rectBoundingBox.maxpoint.z=1.0;
		gtri[i].rectBoundingBox.intervalX.StartValue=1.1;
		gtri[i].rectBoundingBox.intervalX.EndValue=42.1;
		gtri[i].rectBoundingBox.intervalY.StartValue=1.1;
		gtri[i].rectBoundingBox.intervalY.EndValue=42.1;
		gtri[i].rectBoundingBox.intervalZ.StartValue=1.1;
		gtri[i].rectBoundingBox.intervalZ.EndValue=42.1;

	}
	runGPUIntersectionTest(goct,gtri,gindices,NUMOFOCTANTS,62,62);
    return 0;
}

