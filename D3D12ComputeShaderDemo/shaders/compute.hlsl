cbuffer cbCS : register(b0)
{
    int g_constant;
    uint g_minWaveLanes;
};

groupshared int sharedBuffer[1024];

StructuredBuffer<int> srcBuffer: register(t0);      // Shader Resource View (SRV) buffer
RWStructuredBuffer<int> dstBuffer: register(u0);    // Unordered Access View (UAV) buffer
RWStructuredBuffer<int> rwBuffer: register(u1);     // Unordered Access View (UAV) buffer

[numthreads(1024, 1, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    const uint globalIndex = tid.x;
    dstBuffer[globalIndex] = srcBuffer[globalIndex] + g_constant;

    // Do the second calculation...

    // Firstly, put the data into the group-shared memory
    sharedBuffer[groupIndex] = rwBuffer[globalIndex];

    GroupMemoryBarrierWithGroupSync();

    // Use the first g_minWaveLanes threads of each group to calculate the sum
    if (groupIndex < g_minWaveLanes)
    {
        int sum = 0;
        for(uint i = 0; i < 1024; i++)
            sum += sharedBuffer[i];

        rwBuffer[groupID.x] = sum;
    }
}

