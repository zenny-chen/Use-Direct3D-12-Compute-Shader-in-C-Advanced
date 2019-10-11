cbuffer cbCS : register(b0)
{
    int g_constant;
};

groupshared int sharedBuffer[1024];

StructuredBuffer<int> srcBuffer: register(t0);      // SRV
RWStructuredBuffer<int> dstBuffer: register(u0);    // UAV
RWStructuredBuffer<int> rwBuffer: register(u1);     // UAV

[numthreads(1024, 1, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    const uint globalIndex = tid.x;
    dstBuffer[globalIndex] = srcBuffer[globalIndex] + g_constant;

    // Do the second calculation...
    const uint localIndex = localTID.x;

    // Firstly, put the data into the group-shared memory
    sharedBuffer[localIndex] = rwBuffer[globalIndex];

    GroupMemoryBarrierWithGroupSync();

    // Use the first thread of each group to calculate the sum
    if (localIndex == 0)
    {
        int sum = 0;
        for(uint i = 0; i < 1024; i++)
            sum += sharedBuffer[i];

        rwBuffer[groupID.x] = sum;
    }
}

