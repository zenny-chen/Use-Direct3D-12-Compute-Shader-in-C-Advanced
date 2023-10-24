#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdalign.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

enum
{
    // Max number of hardware adapter count
    MAX_HARDWARE_ADAPTER_COUNT = 16,

    // Test data element count
    TEST_DATA_COUNT = 4096,

    // Command Queue Fence signal value for data transfer
    COPY_SYNC_SIGNAL_VALUE = 1,

    // Command Queue Fence signal value for compute shader execution
    COMPUTE_SYNC_SIGNAL_VALUE = 2
};

// The factory used to create D3D12 devices
static IDXGIFactory4* s_factory;

// The compatible D3D12 device object
static ID3D12Device *s_device;

// The root signature for compute pipeline state object
static ID3D12RootSignature *s_computeRootSignature;

// The compute pipeline state object
static ID3D12PipelineState *s_computeState;

// The descriptor heap resource object. 
// In this sample, there're two slots in this heap. 
// The first slot stores the shader view resource descriptor, 
// and the second slot stores the unordered access view descriptor.
static ID3D12DescriptorHeap* s_heap;

// The destination buffer object with unordered access view type
static ID3D12Resource *s_dstDataBuffer;

// The source buffer object with shader source view type
static ID3D12Resource *s_srcDataBuffer;

// The intermediate buffer object used to copy the source data to the SRV buffer
static ID3D12Resource* s_uploadBuffer;

// The second destination buffer object with unordered access view type
static ID3D12Resource* s_dst2Buffer;

// The intermediate buffer object used to copy specified data to the the second destination data
static ID3D12Resource* s_dst2UploadBuffer;

// The constant buffer object
static ID3D12Resource* s_constantBuffer;

// The intermediate buffer object used to upload data to the constant buffer object
static ID3D12Resource* s_constantUploadBuffer;

// The heap descriptor(of SRV, UAV and CBV type)  size
static size_t s_srvUavDescriptorSize;

// The command allocator object
static ID3D12CommandAllocator *s_computeAllocator;

// The command queue object
static ID3D12CommandQueue *s_computeCommandQueue;

// The command list object
static ID3D12GraphicsCommandList *s_computeCommandList;

// Fence object
static ID3D12Fence* s_fence;

// Win32 API event handle
static HANDLE s_hEvent;

// Indicate whether the specified D3D device supports root signature version 1.1 or not
static bool s_supportSignatureVersion1_1;

// The first source data buffer
static int *s_dataBuffer0;

// The second source data buffer
static int* s_dataBuffer1;


static void TransWStrToString(char dstBuf[], const WCHAR srcBuf[])
{
    if (dstBuf == NULL || srcBuf == NULL) return;

    const int len = WideCharToMultiByte(CP_UTF8, 0, srcBuf, -1, NULL, 0, NULL, NULL);
    WideCharToMultiByte(CP_UTF8, 0, srcBuf, -1, dstBuf, len, NULL, NULL);
    dstBuf[len] = '\0';
}

static D3D12_SHADER_BYTECODE CreateCompiledShaderObjectFromPath(const char csoPath[])
{
    D3D12_SHADER_BYTECODE result = { 0 };
    FILE* fp = NULL;
    const errno_t err = fopen_s(&fp, csoPath, "rb");
    if (err != 0 || fp == NULL)
    {
        fprintf(stderr, "Read compiled shader object file: `%s` failed: %d\n", csoPath, err);
        if (fp != NULL) {
            fclose(fp);
        }
        return result;
    }

    fseek(fp, 0, SEEK_END);
    const size_t fileSize = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    const size_t codeElemCount = (fileSize + sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t* csoBlob = (uint32_t*)calloc(codeElemCount, sizeof(uint32_t));
    if (csoBlob == NULL)
    {
        fprintf(stderr, "Lack of system memory to allocate memory for `%s` CSO object!\n", csoPath);
        return result;
    }
    if (fread(csoBlob, 1, fileSize, fp) < 1) {
        printf("WARNING: Read compiled shader object file `%s` error!\n", csoPath);
    }
    fclose(fp);

    result.pShaderBytecode = csoBlob;
    result.BytecodeLength = fileSize;
    return result;
}

static bool QueryDeviceSupportedMaxFeatureLevel(void)
{
    const D3D_FEATURE_LEVEL requestedLevels[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2
    };
    D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {
        .NumFeatureLevels = sizeof(requestedLevels) / sizeof(requestedLevels[0]),
        .pFeatureLevelsRequested = requestedLevels
    };

    auto const hRes = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_FEATURE_LEVELS, &featureLevels, sizeof(featureLevels));
    if (FAILED(hRes))
    {
        fprintf(stderr, "CheckFeatureSupport for `D3D12_FEATURE_FEATURE_LEVELS` failed: %ld\n", hRes);
        return false;
    }

    const D3D_FEATURE_LEVEL maxFeatureLevel = featureLevels.MaxSupportedFeatureLevel;

    char strBuf[32] = { '\0' };
    switch (maxFeatureLevel)
    {
    case D3D_FEATURE_LEVEL_1_0_CORE:
        strcpy_s(strBuf, sizeof(strBuf), "1.0 core");
        break;

    case D3D_FEATURE_LEVEL_9_1:
        strcpy_s(strBuf, sizeof(strBuf), "9.1");
        break;

    case D3D_FEATURE_LEVEL_9_2:
        strcpy_s(strBuf, sizeof(strBuf), "9.2");
        break;

    case D3D_FEATURE_LEVEL_9_3:
        strcpy_s(strBuf, sizeof(strBuf), "9.3");
        break;

    case D3D_FEATURE_LEVEL_10_0:
        strcpy_s(strBuf, sizeof(strBuf), "10.0");
        break;

    case D3D_FEATURE_LEVEL_10_1:
        strcpy_s(strBuf, sizeof(strBuf), "10.1");
        break;

    case D3D_FEATURE_LEVEL_11_0:
        strcpy_s(strBuf, sizeof(strBuf), "11.0");
        break;

    case D3D_FEATURE_LEVEL_11_1:
        strcpy_s(strBuf, sizeof(strBuf), "11.1");
        break;

    case D3D_FEATURE_LEVEL_12_0:
        strcpy_s(strBuf, sizeof(strBuf), "12.0");
        break;

    case D3D_FEATURE_LEVEL_12_1:
        strcpy_s(strBuf, sizeof(strBuf), "12.1");
        break;

    case D3D_FEATURE_LEVEL_12_2:
        strcpy_s(strBuf, sizeof(strBuf), "12.2");
        break;

    default:
        break;
    }

    printf("Current device supports max feature level: %s\n", strBuf);
    return true;
}

static bool CreateD3D12Device(void)
{
    HRESULT hRes = S_OK;

#if defined(DEBUG) || defined(_DEBUG)
    // In debug mode
    ID3D12Debug* debugController = NULL;
    hRes = D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&debugController);
    if (SUCCEEDED(hRes)) {
        debugController->lpVtbl->EnableDebugLayer(debugController);
    }
    else {
        printf("WARNING: Failed to enable debug layer: %ld\n", hRes);
    }
#endif // defined(DEBUG) || defined(_DEBUG)

    hRes = CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&s_factory);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateDXGIFactory1 failed: %ld\n", hRes);
        return false;
    }

    // Enumerate the adapters (video cards)
    IDXGIAdapter1* hardwareAdapters[MAX_HARDWARE_ADAPTER_COUNT] = { 0 };
    UINT foundAdapterCount;
    for (foundAdapterCount = 0; foundAdapterCount < MAX_HARDWARE_ADAPTER_COUNT; ++foundAdapterCount)
    {
        hRes = s_factory->lpVtbl->EnumAdapters1(s_factory, foundAdapterCount, &hardwareAdapters[foundAdapterCount]);
        if (FAILED(hRes))
        {
            if (hRes != DXGI_ERROR_NOT_FOUND) {
                printf("WARNING: Some error occurred during enumerating adapters: %ld\n", hRes);
            }
            break;
        }
    }
    if (foundAdapterCount == 0)
    {
        fprintf(stderr, "There are no Direct3D capable adapters found on the current platform...\n");
        return false;
    }

    printf("Found %u Direct3D capable device%s in all.\n", foundAdapterCount, foundAdapterCount > 1 ? "s" : "");

    DXGI_ADAPTER_DESC1 adapterDesc = { 0 };
    char strBuf[512] = { '\0' };
    for (UINT i = 0; i < foundAdapterCount; ++i)
    {
        hRes = hardwareAdapters[i]->lpVtbl->GetDesc1(hardwareAdapters[i], &adapterDesc);
        if (FAILED(hRes))
        {
            fprintf(stderr, "hardwareAdapters[%u] GetDesc1 failed: %ld\n", i, hRes);
            return false;
        }

        TransWStrToString(strBuf, adapterDesc.Description);
        printf("Adapter[%u]: %s\n", i, strBuf);
    }
    printf("Please Choose which adapter to use: ");

    gets_s(strBuf, sizeof(strBuf));

    char* endChar = NULL;
    int selectedAdapterIndex = atoi(strBuf);
    if (selectedAdapterIndex < 0 || selectedAdapterIndex >= (int)foundAdapterCount)
    {
        puts("WARNING: The index you input exceeds the range of available adatper count. So adatper[0] will be used!");
        selectedAdapterIndex = 0;
    }

    hRes = hardwareAdapters[selectedAdapterIndex]->lpVtbl->GetDesc1(hardwareAdapters[selectedAdapterIndex], &adapterDesc);
    if (FAILED(hRes))
    {
        fprintf(stderr, "hardwareAdapters[%d] GetDesc1 failed: %ld\n", selectedAdapterIndex, hRes);
        return false;
    }

    TransWStrToString(strBuf, adapterDesc.Description);

    printf("\nYou have chosen adapter[%ld]\n", selectedAdapterIndex);
    printf("Adapter description: %s\n", strBuf);
    printf("Dedicated Video Memory: %.1f GB\n", (double)(adapterDesc.DedicatedVideoMemory) / (1024.0 * 1024.0 * 1024.0));
    printf("Dedicated System Memory: %.1f GB\n", (double)(adapterDesc.DedicatedSystemMemory) / (1024.0 * 1024.0 * 1024.0));
    printf("Shared System Memory: %.1f GB\n", (double)(adapterDesc.SharedSystemMemory) / (1024.0 * 1024.0 * 1024.0));

    hRes = D3D12CreateDevice((IUnknown*)hardwareAdapters[selectedAdapterIndex], D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void**)&s_device);
    if (FAILED(hRes))
    {
        fprintf(stderr, "D3D12CreateDevice failed: %ld\n", hRes);
        return false;
    }

    if(!QueryDeviceSupportedMaxFeatureLevel()) return false;

    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { .HighestShaderModel = D3D_HIGHEST_SHADER_MODEL };
    hRes = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
    if (FAILED(hRes))
    {
        fprintf(stderr, "CheckFeatureSupport for `D3D12_FEATURE_SHADER_MODEL` failed: %ld\n", hRes);
        return false;
    }

    const int minor = shaderModel.HighestShaderModel & 0x0f;
    const int major = shaderModel.HighestShaderModel >> 4;
    printf("Current device support highest shader model: %d.%d\n", major, minor);

    D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignature = { .HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1 };
    hRes = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_ROOT_SIGNATURE, &rootSignature, sizeof(rootSignature));
    if (FAILED(hRes))
    {
        fprintf(stderr, "CheckFeatureSupport for `D3D12_FEATURE_DATA_ROOT_SIGNATURE` failed: %ld\n", hRes);
        return false;
    }

    const char* signatureVersion = "1.0";
    switch (rootSignature.HighestVersion)
    {
    case D3D_ROOT_SIGNATURE_VERSION_1_0:
    default:
        s_supportSignatureVersion1_1 = false;
        break;

    case D3D_ROOT_SIGNATURE_VERSION_1_1:
        signatureVersion = "1.1";
        s_supportSignatureVersion1_1 = true;
        break;
    }
    printf("Current device supports highest root signature version: %s\n", signatureVersion);

    puts("\n================================================\n");

    return true;
}

static bool CreateRootSignature(void)
{
    const D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
                                    D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                                    D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                                    D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                                    D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS |
                                    D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS;

    ID3DBlob* errorBlob = NULL;
    ID3DBlob* signature = NULL;
    HRESULT hRes = S_OK;
    if (s_supportSignatureVersion1_1)
    {
        const D3D12_DESCRIPTOR_RANGE1 ranges[] = {
            // t0
            {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 0,
                .RegisterSpace = 0,
                .Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            },
            // u0
            {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 0,
                .RegisterSpace = 0,
                .Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            },
            // u1
            {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 1,
                .RegisterSpace = 0,
                .Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            }
        };

        // There're 3 parameters which will be passed to the compute shader
        const D3D12_ROOT_PARAMETER1 rootParameters[] = {
            // The first is the constant buffer object, b0
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV,
                .Descriptor = {
                    .ShaderRegister = 0,
                    .RegisterSpace = 0,
                    .Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC
                },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            },
            // The second is the shader source view object, t0
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[0] },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            },
            // The third is the unordered access view object, u0
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[1] },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            },
            // The fourth is the unordered access view object, u1
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[2] },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            }
        };

        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc = {
            .Version = D3D_ROOT_SIGNATURE_VERSION_1_1,
            .Desc_1_1 = {
                .NumParameters = sizeof(rootParameters) / sizeof(rootParameters[0]),
                .pParameters = rootParameters,
                .NumStaticSamplers = 0,
                .pStaticSamplers = NULL,
                .Flags = rootSignatureFlags
            }
        };

        hRes = D3D12SerializeVersionedRootSignature(&computeRootSignatureDesc, &signature, &errorBlob);
    }
    else
    {
        // D3D_ROOT_SIGNATURE_VERSION_1_0 situation
        const D3D12_DESCRIPTOR_RANGE ranges[] = {
            // t0
            {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 0,
                .RegisterSpace = 0,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            },
            // u0
            {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 0,
                .RegisterSpace = 0,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            },
            // u1
            {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 1,
                .RegisterSpace = 0,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            }
        };

        // There're 3 parameters which will be passed to the compute shader
        const D3D12_ROOT_PARAMETER rootParameters[] = {
            // The first is the constant buffer object, b0
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV,
                .Descriptor = {
                    .ShaderRegister = 0,
                    .RegisterSpace = 0,
                },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            },
            // The second is the shader source view object, t0
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[0] },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            },
            // The third is the unordered access view object, u0
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[1] },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            },
            // The fourth is the unordered access view object, u1
            {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[2] },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
            }
        };

        const D3D12_ROOT_SIGNATURE_DESC computeRootSignatureDesc = {
            .NumParameters = (UINT)(sizeof(rootParameters) / sizeof(rootParameters[0])),
            .pParameters = rootParameters,
            .NumStaticSamplers = 0,
            .Flags = rootSignatureFlags
        };

        hRes = D3D12SerializeRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &errorBlob);
    }

    do
    {
        if (FAILED(hRes))
        {
            fprintf(stderr, "D3D12SerializeVersionedRootSignature failed: %ld\n", hRes);
            break;
        }

        hRes = s_device->lpVtbl->CreateRootSignature(s_device, 0, signature->lpVtbl->GetBufferPointer(signature),
            signature->lpVtbl->GetBufferSize(signature), &IID_ID3D12RootSignature, &s_computeRootSignature);
        if (FAILED(hRes))
        {
            fprintf(stderr, "CreateRootSignature failed: %ld\n", hRes);
            break;
        }
    }
    while (false);

    if (errorBlob != NULL) {
        errorBlob->lpVtbl->Release(errorBlob);
    }
    if (signature != NULL) {
        signature->lpVtbl->Release(signature);
    }

    if (FAILED(hRes)) return false;

    // This setting is optional.
    hRes = s_computeRootSignature->lpVtbl->SetName(s_computeRootSignature, L"s_computeRootSignature");
    if (FAILED(hRes))
    {
        fprintf(stderr, "s_computeRootSignature setName failed: %ld\n", hRes);
        return false;
    }

    return true;
}

// Updates subresources, all the subresource arrays should be populated.
// This function is the C-style implementation translated from C++ style inline function in the D3DX12 library.
static void WriteDeviceResourceAndSync(
    _In_ ID3D12GraphicsCommandList* commandList,
    _In_ ID3D12Resource* pDestinationDeviceResource,
    _In_ ID3D12Resource* pUploadHostResource,
    size_t dstOffset,
    size_t srcoffset,
    size_t dataSize,
    bool isDstReadWrite)
{
    const D3D12_RESOURCE_BARRIER beginCopyBarrier = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = pDestinationDeviceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COMMON,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_DEST
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, 1, &beginCopyBarrier);

    commandList->lpVtbl->CopyBufferRegion(commandList, pDestinationDeviceResource, (UINT64)dstOffset, pUploadHostResource, srcoffset, dataSize);

    const D3D12_RESOURCE_STATES hasUAVState = isDstReadWrite ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON;
    const D3D12_RESOURCE_BARRIER endCopyBarrier = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = pDestinationDeviceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST,
            .StateAfter = hasUAVState
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, 1, &endCopyBarrier);
}

static void SyncAndReadDeviceResources(
    _In_ ID3D12GraphicsCommandList* commandList,
    _In_ ID3D12Resource* pReadbackHostResource1,
    _In_ ID3D12Resource* pSourceDeviceResource1,
    _In_ ID3D12Resource* pReadbackHostResource2,
    _In_ ID3D12Resource* pSourceDeviceResource2)
{
    const D3D12_RESOURCE_BARRIER beginCopyBarriers[] = {
        {
            .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
            .Transition = {
                .pResource = pSourceDeviceResource1,
                .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE
            }
        },
        {
            .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
            .Transition = {
                .pResource = pSourceDeviceResource2,
                .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE
            }
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, sizeof(beginCopyBarriers) / sizeof(beginCopyBarriers[0]), beginCopyBarriers);

    commandList->lpVtbl->CopyResource(commandList, pReadbackHostResource1, pSourceDeviceResource1);
    commandList->lpVtbl->CopyResource(commandList, pReadbackHostResource2, pSourceDeviceResource2);

    const D3D12_RESOURCE_BARRIER endCopyBarriers[] = {
        {
            .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
            .Transition = {
                .pResource = pSourceDeviceResource1,
                .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE,
                .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            }
        },
        {
            .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
            .Transition = {
                .pResource = pSourceDeviceResource2,
                .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE,
                .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            }
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, sizeof(endCopyBarriers) / sizeof(endCopyBarriers[0]), endCopyBarriers);
}

// Create the write-only Shader Resource View buffer object
static ID3D12Resource* CreateSRVBuffer(const void* inputData, size_t dataSize, UINT elemCount, UINT elemSize)
{
    ID3D12Resource *resultBuffer = NULL;
    HRESULT hr = S_OK;

    do
    {
        const D3D12_HEAP_PROPERTIES heapProperties = {
            .Type = D3D12_HEAP_TYPE_DEFAULT,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };
        const D3D12_HEAP_PROPERTIES heapUploadProperties = {
            .Type = D3D12_HEAP_TYPE_UPLOAD,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };

        const D3D12_RESOURCE_DESC resourceDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = { .Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_NONE
        };
        const D3D12_RESOURCE_DESC uploadBufferDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = { .Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_NONE
        };

        // Create the SRV buffer and make it as the copy destination.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                                    D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void**)&resultBuffer);
        if (FAILED(hr))
        {
            fprintf(stderr, "CreateCommittedResource for resultBuffer failed: %ld\n", hr);
            break;
        }

        // Create the upload buffer and make it as the generic read intermediate.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapUploadProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void**)&s_uploadBuffer);
        if (FAILED(hr))
        {
            fprintf(stderr, "CreateCommittedResource for s_uploadBuffer failed: %ld\n", hr);
            break;
        }

        // Transfer data from host to the device SRV buffer
        void* hostMemPtr = NULL;
        const D3D12_RANGE readRange = { 0, 0 };
        hr = s_uploadBuffer->lpVtbl->Map(s_uploadBuffer, 0, &readRange, &hostMemPtr);
        if (FAILED(hr))
        {
            fprintf(stderr, "Map s_uploadBuffer failed: %ld\n", hr);
            break;
        }

        memcpy(hostMemPtr, inputData, dataSize);
        s_uploadBuffer->lpVtbl->Unmap(s_uploadBuffer, 0, NULL);

        // Upload data from s_uploadBuffer to resultBuffer
        WriteDeviceResourceAndSync(s_computeCommandList, resultBuffer, s_uploadBuffer, 0U, 0U, dataSize, false);

        // Attention! None of the operations above has been executed.
        // They have just been put into the command list.
        // So the intermediate buffer s_uploadBuffer MUST NOT be released here.

        // Setup the SRV descriptor. This will be stored in the first slot of the heap.
        const D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {
            .Format = DXGI_FORMAT_UNKNOWN,
            .ViewDimension = D3D12_SRV_DIMENSION_BUFFER,
            .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            .Buffer = {
                .FirstElement = 0,
                .NumElements = elemCount,
                .StructureByteStride = elemSize,
                .Flags = D3D12_BUFFER_SRV_FLAG_NONE
            }
        };

        // Get the descriptor handle from the descriptor heap.
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
        s_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_heap, &srvHandle);

        // Create the SRV for the buffer with the descriptor handle
        s_device->lpVtbl->CreateShaderResourceView(s_device, resultBuffer, &srvDesc, srvHandle);
    }
    while (false);

    if (FAILED(hr))
    {
        fprintf(stderr, "CreateSRVBuffer failed: %ld\n", hr);
        return NULL;
    }

    return resultBuffer;
}

// Create the write-only Unordered Access View buffer object for the first destination buffer object
static ID3D12Resource* CreateUAV_RBuffer(const void* inputData, size_t dataSize, UINT elemCount, UINT elemSize)
{
    ID3D12Resource *resultBuffer = NULL;
    HRESULT hr = S_OK;

    do
    {
        const D3D12_HEAP_PROPERTIES heapProperties = {
            .Type = D3D12_HEAP_TYPE_DEFAULT,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };
        const D3D12_RESOURCE_DESC resourceDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1, 
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };

        // Create the UAV buffer and make it in the unordered access state.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                                    D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void**)&resultBuffer);

        if (FAILED(hr))
        {
            fprintf(stderr, "Failed to create resultBuffer: %ld\n", hr);
            return NULL;
        }

        // Setup the UAV descriptor. This will be stored in the second slot of the heap.
        const D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {
            .Format = DXGI_FORMAT_UNKNOWN,
            .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
            .Buffer = {
                .FirstElement = 0,
                .NumElements = elemCount,
                .StructureByteStride = elemSize,
                .CounterOffsetInBytes = 0,
                .Flags = D3D12_BUFFER_UAV_FLAG_NONE
            }
        };

        // Get the descriptor handle from the descriptor heap.
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle;
        s_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_heap, &uavHandle);

        // It will occupy the second slot.
        uavHandle.ptr += 1U * s_srvUavDescriptorSize;

        s_device->lpVtbl->CreateUnorderedAccessView(s_device, resultBuffer, NULL, &uavDesc, uavHandle);
    }
    while (false);

    if (FAILED(hr))
    {
        fprintf(stderr, "Create UAV Buffer failed: %ld\n", hr);
        return NULL;
    }

    return resultBuffer;
}

// Create the read-write Unordered Access View buffer object for the second destination buffer object
static bool CreateUAV2_RWBuffer(const void* inputData, size_t dataSize, UINT elemCount, UINT elemSize)
{
    HRESULT hr = S_OK;

    do
    {
        const D3D12_HEAP_PROPERTIES heapProperties = {
            .Type = D3D12_HEAP_TYPE_DEFAULT,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };
        const D3D12_HEAP_PROPERTIES heapUploadProperties = {
            .Type = D3D12_HEAP_TYPE_UPLOAD,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };

        const D3D12_RESOURCE_DESC resourceDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };
        const D3D12_RESOURCE_DESC uploadBufferDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_NONE
        };

        // Create the UAV buffer and make it in the unordered access state.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                                    D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void**)&s_dst2Buffer);
        if (FAILED(hr))
        {
            fprintf(stderr, "Failed to create resultBuffer: %ld\n", hr);
            break;
        }

        // Create the upload buffer and make it as the generic read intermediate.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapUploadProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void**)&s_dst2UploadBuffer);
        if (FAILED(hr))
        {
            fprintf(stderr, "CreateCommittedResource for s_dst2UploadBuffer failed: %ld\n", hr);
            break;
        }

        // Transfer data from host to the device UAV buffer
        void* hostMemPtr = NULL;
        const D3D12_RANGE readRange = { 0, 0 };
        hr = s_dst2UploadBuffer->lpVtbl->Map(s_dst2UploadBuffer, 0, &readRange, &hostMemPtr);
        if (FAILED(hr))
        {
            fprintf(stderr, "Map s_dst2UploadBuffer failed: %ld\n", hr);
            break;
        }

        memcpy(hostMemPtr, inputData, dataSize);
        s_dst2UploadBuffer->lpVtbl->Unmap(s_dst2UploadBuffer, 0, NULL);

        // Upload data from s_dst2UploadBuffer to s_dst2Buffer
        WriteDeviceResourceAndSync(s_computeCommandList, s_dst2Buffer, s_dst2UploadBuffer, 0U, 0U, dataSize, true);

        // Setup the UAV descriptor. This will be stored in the second slot of the heap.
        const D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {
            .Format = DXGI_FORMAT_UNKNOWN,
            .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
            .Buffer = {
                .FirstElement = 0,
                .NumElements = elemCount,
                .StructureByteStride = elemSize,
                .CounterOffsetInBytes = 0,
                .Flags = D3D12_BUFFER_UAV_FLAG_NONE
            }
        };

        // Get the descriptor handle from the descriptor heap.
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle;
        s_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_heap, &uavHandle);

        // It will occupy the third slot.
        uavHandle.ptr += 2U * s_srvUavDescriptorSize;

        s_device->lpVtbl->CreateUnorderedAccessView(s_device, s_dst2Buffer, NULL, &uavDesc, uavHandle);
    }
    while (false);

    if (FAILED(hr))
    {
        fprintf(stderr, "Create UAV Buffer failed: %ld\n", hr);
        return false;
    }

    return true;
}

// Create and initialize the constant buffer object
static bool CreateConstantBuffer(const void* inputData, size_t dataSize)
{
    const D3D12_HEAP_PROPERTIES heapProperties = {
        .Type = D3D12_HEAP_TYPE_DEFAULT,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };
    const D3D12_HEAP_PROPERTIES heapUploadProperties = {
        .Type = D3D12_HEAP_TYPE_UPLOAD,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };

    const D3D12_RESOURCE_DESC resourceDesc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = dataSize,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_NONE
    };
    const D3D12_RESOURCE_DESC uploadBufferDesc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = dataSize,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_NONE
    };

    // Create the constant buffer and make it as the copy destination.
    HRESULT hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                                        D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void**)&s_constantBuffer);
    if (FAILED(hr))
    {
        fprintf(stderr, "CreateCommittedResource for s_constantBuffer failed: %ld\n", hr);
        return false;
    }

    // Create the upload buffer and make it as the generic read intermediate.
    hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapUploadProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void**)&s_constantUploadBuffer);
    if (FAILED(hr))
    {
        fprintf(stderr, "CreateCommittedResource for s_constantUploadBuffer failed: %ld\n", hr);
        return false;
    }

    // This setting is optional.
    s_constantBuffer->lpVtbl->SetName(s_constantBuffer, L"s_constantBuffer");

    // Transfer data from host to the device UAV buffer
    void* hostMemPtr = NULL;
    const D3D12_RANGE readRange = { 0, 0 };
    hr = s_constantUploadBuffer->lpVtbl->Map(s_constantUploadBuffer, 0, &readRange, &hostMemPtr);
    if (FAILED(hr))
    {
        fprintf(stderr, "Map s_constantUploadBuffer failed: %ld\n", hr);
        return false;
    }

    memcpy(hostMemPtr, inputData, dataSize);
    s_constantUploadBuffer->lpVtbl->Unmap(s_constantUploadBuffer, 0, NULL);

    // Upload data from s_constantUploadBuffer to s_constantBuffer
    WriteDeviceResourceAndSync(s_computeCommandList, s_constantBuffer, s_constantUploadBuffer, 0U, 0U, dataSize, false);

    return true;
}

// Create the compute pipeline state object
static bool CreateComputePipelineStateObject(void)
{
    // ---- Create descriptor heaps. ----
    const D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {
        // There are three descriptors for the heap. One for SRV buffer, the other two for UAV buffers
        .NumDescriptors = 3,
        .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        .NodeMask = 0
    };

    HRESULT hr = s_device->lpVtbl->CreateDescriptorHeap(s_device, &srvUavHeapDesc, &IID_ID3D12DescriptorHeap, (void**)&s_heap);
    if (FAILED(hr))
    {
        fprintf(stderr, "Failed to create s_srvHeap: %ld\n", hr);
        return false;
    }

    // This setting is optional.
    s_heap->lpVtbl->SetName(s_heap, L"s_heap");

    // Get the size of each descriptor handle
    s_srvUavDescriptorSize = s_device->lpVtbl->GetDescriptorHandleIncrementSize(s_device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // ---- Load Assets ----
    const D3D12_SHADER_BYTECODE computeShaderObj = CreateCompiledShaderObjectFromPath("shaders/compute.cso");
    if (computeShaderObj.pShaderBytecode == NULL || computeShaderObj.BytecodeLength == 0) return false;

    // Describe and create the compute pipeline state object (PSO).
    const D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {
        .pRootSignature = s_computeRootSignature,
        .CS = computeShaderObj,
        .NodeMask = 0,
        .CachedPSO = {.pCachedBlob = NULL, .CachedBlobSizeInBytes = 0 },
        .Flags = D3D12_PIPELINE_STATE_FLAG_NONE
    };
    hr = s_device->lpVtbl->CreateComputePipelineState(s_device, &computePsoDesc, &IID_ID3D12PipelineState, (void**)&s_computeState);
    if (FAILED(hr)) return false;

    return true;
}

// Initialize the command list and the command queue
static bool InitComputeCommands(void)
{
    const D3D12_COMMAND_QUEUE_DESC queueDesc = {
        .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
        .Priority = 0,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0
    };
    HRESULT hRes = s_device->lpVtbl->CreateCommandQueue(s_device, &queueDesc, &IID_ID3D12CommandQueue, (void**)&s_computeCommandQueue);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateCommandQueue failed: %ld\n", hRes);
        return false;
    }

    hRes = s_device->lpVtbl->CreateCommandAllocator(s_device, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&s_computeAllocator);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateCommandAllocator failed: %ld\n", hRes);
        return false;
    }

    hRes = s_device->lpVtbl->CreateCommandList(s_device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_computeAllocator, NULL, &IID_ID3D12CommandList, (void**)&s_computeCommandList);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateCommandList failed: %ld\n", hRes);
        return false;
    }

    return true;
}

// Create the source buffer object and the destination buffer object.
// Initialize the SRV buffer object with the input buffer
static bool CreateBuffers(void)
{
    enum { bufferSize = TEST_DATA_COUNT * sizeof(*s_dataBuffer0) };

    // Allocate the source data buffers
    s_dataBuffer0 = malloc(bufferSize);
    s_dataBuffer1 = malloc(bufferSize);
    if (s_dataBuffer0 == NULL || s_dataBuffer1 == NULL)
    {
        fprintf(stderr, "Lack of memory for host buffers...\n");
        return false;
    }

    // Initialize the source data buffers
    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        s_dataBuffer0[i] = i + 1;
    }

    int index = 0;
    enum { nGroups = TEST_DATA_COUNT / 1024 };
    for (int i = 0; i < nGroups; i++)
    {
        for (int j = 0; j < 1024; j++) {
            s_dataBuffer1[index++] = i + 1;
        }
    }

    // Create the compute shader's constant buffer.
    s_srcDataBuffer = CreateSRVBuffer(s_dataBuffer0, bufferSize, TEST_DATA_COUNT, (UINT)sizeof(int));
    s_dstDataBuffer = CreateUAV_RBuffer(NULL, bufferSize, TEST_DATA_COUNT, (UINT)sizeof(int));
    if (!CreateUAV2_RWBuffer(s_dataBuffer1, bufferSize, TEST_DATA_COUNT, (UINT)sizeof(int))) return false;

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = { 0 };
    HRESULT hr = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_D3D12_OPTIONS1,
                                                        &options1, sizeof(options1));
    if (FAILED(hr)) return false;

    struct { int cbValue; UINT minWaveLanes; } cbuffer = {
        1, 64
    };

    if (options1.WaveOps)
    {
        puts("Current GPU supports HLSL 6.0 wave operations!!");
        printf("The minimum wave lane count is: %u\n", options1.WaveLaneCountMin);

        cbuffer.minWaveLanes = options1.WaveLaneCountMin;
    }

    return CreateConstantBuffer(&cbuffer, sizeof(cbuffer));
}

// Create ID3D12Fence fence object and Win32 s_hEvent handle
static bool CreateFenceAndEvent(void)
{
    HRESULT hRes = s_device->lpVtbl->CreateFence(s_device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&s_fence);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateFence failed: %ld\n", hRes);
        return false;
    }

    s_hEvent = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (s_hEvent == NULL)
    {
        const DWORD err = GetLastError();
        fprintf(stderr, "Failed to create event handle: %u\n", err);
        return false;
    }

    return true;
}

// Wait for the whole command queue completed
static void SyncCommandQueue(ID3D12CommandQueue* commandQueue, ID3D12Device* device, UINT64 signalValue)
{
    // Add an instruction to the command queue to set a new fence point.  Because we 
    // are on the GPU timeline, the new fence point won't be set until the GPU finishes
    // processing all the commands prior to this Signal().
    HRESULT hRes = commandQueue->lpVtbl->Signal(commandQueue, s_fence, signalValue);
    if (FAILED(hRes)) {
        fprintf(stderr, "Signal failed: %ld\n", hRes);
    }

    // Wait until the GPU has completed commands up to this fence point.
    // Fire event when GPU hits current fence.  
    hRes = s_fence->lpVtbl->SetEventOnCompletion(s_fence, signalValue, s_hEvent);
    if (FAILED(hRes)) {
        fprintf(stderr, "Set event failed: %ld\n", hRes);
    }

    // Wait until the GPU hits current fence event is fired.
    WaitForSingleObject(s_hEvent, INFINITE);
}

// Do the compute operation and fetch the result
static void DoCompute(void)
{
    ID3D12Resource *readBackBuffer = NULL;
    ID3D12Resource* readBackBuffer2 = NULL;

    const D3D12_HEAP_PROPERTIES heapProperties = {
        .Type = D3D12_HEAP_TYPE_READBACK,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };
    const D3D12_RESOURCE_DESC resourceDesc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = TEST_DATA_COUNT * sizeof(*s_dataBuffer0),
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_NONE
    };

    // Source and Destination buffer resource must have the same size/width,
    // So the resourceDesc2 MUST NOT set the width that is not equal to `TEST_DATA_COUNT * sizeof(*s_dataBuffer0)`
    const D3D12_RESOURCE_DESC resourceDesc2 = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = TEST_DATA_COUNT * sizeof(*s_dataBuffer1),
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_NONE
    };

    // Create the read-back buffer object that will fetch the result from the UAV buffer object.
    // And make it as the copy destination.
    HRESULT hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&readBackBuffer);
    if (FAILED(hr)) return;

    hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc2,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&readBackBuffer2);
    if (FAILED(hr)) return;

    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished execution on the GPU.
    hr = s_computeAllocator->lpVtbl->Reset(s_computeAllocator);
    if(FAILED(hr)) return;

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    hr = s_computeCommandList->lpVtbl->Reset(s_computeCommandList, s_computeAllocator, s_computeState);
    if(FAILED(hr)) return;

    s_computeCommandList->lpVtbl->SetComputeRootSignature(s_computeCommandList, s_computeRootSignature);

    ID3D12DescriptorHeap* ppHeaps[] = { s_heap };
    s_computeCommandList->lpVtbl->SetDescriptorHeaps(s_computeCommandList, sizeof(ppHeaps) / sizeof(ppHeaps[0]), ppHeaps);

    D3D12_GPU_DESCRIPTOR_HANDLE srvHandle;
    // Get the SRV GPU descriptor handle from the descriptor heap
    s_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_heap, &srvHandle);

    D3D12_GPU_DESCRIPTOR_HANDLE uavHandle, uavHandle2;
    // Get the UAV GPU descriptor handle from the descriptor heap
    s_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_heap, &uavHandle);
    uavHandle.ptr += 1U * s_srvUavDescriptorSize;

    s_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_heap, &uavHandle2);
    uavHandle2.ptr += 2U * s_srvUavDescriptorSize;

    // Setup the input parameters
    s_computeCommandList->lpVtbl->SetComputeRootConstantBufferView(s_computeCommandList, 0, s_constantBuffer->lpVtbl->GetGPUVirtualAddress(s_constantBuffer));
    s_computeCommandList->lpVtbl->SetComputeRootDescriptorTable(s_computeCommandList, 1, srvHandle);
    s_computeCommandList->lpVtbl->SetComputeRootDescriptorTable(s_computeCommandList, 2, uavHandle);
    s_computeCommandList->lpVtbl->SetComputeRootDescriptorTable(s_computeCommandList, 3, uavHandle2);

    // Dispatch the GPU threads
    s_computeCommandList->lpVtbl->Dispatch(s_computeCommandList, 4, 1, 1);

    // Sync the compute shader execution and transfer the dst buffers to readback buffers
    SyncAndReadDeviceResources(s_computeCommandList, readBackBuffer, s_dstDataBuffer, readBackBuffer2, s_dst2Buffer);

    // Close the command list
    s_computeCommandList->lpVtbl->Close(s_computeCommandList);

    s_computeCommandQueue->lpVtbl->ExecuteCommandLists(s_computeCommandQueue, 1, 
                                                    (ID3D12CommandList* const[]) { (ID3D12CommandList*)s_computeCommandList });

    SyncCommandQueue(s_computeCommandQueue, s_device, COMPUTE_SYNC_SIGNAL_VALUE);

    void* pData = NULL;
    D3D12_RANGE range = { 0, TEST_DATA_COUNT };
    // Map the memory buffer so that we may access the data from the host side.
    hr = readBackBuffer->lpVtbl->Map(readBackBuffer, 0, &range, &pData);
    if (FAILED(hr)) return;

    int* resultBuffer = malloc(TEST_DATA_COUNT * sizeof(*resultBuffer));
    if (resultBuffer == NULL) return;
    memcpy(resultBuffer, pData, TEST_DATA_COUNT * sizeof(*resultBuffer));

    // After copying the data, just release the read-back buffer object.
    readBackBuffer->lpVtbl->Unmap(readBackBuffer, 0, NULL);
    readBackBuffer->lpVtbl->Release(readBackBuffer);

    int* resultBuffer2 = malloc(TEST_DATA_COUNT * sizeof(*resultBuffer2));
    if (resultBuffer2 == NULL) return;
    range = (D3D12_RANGE){ 0, TEST_DATA_COUNT };
    hr = readBackBuffer2->lpVtbl->Map(readBackBuffer2, 0, &range, &pData);
    if (FAILED(hr)) return;

    memcpy(resultBuffer2, pData, TEST_DATA_COUNT * sizeof(*resultBuffer2));

    readBackBuffer2->lpVtbl->Unmap(readBackBuffer2, 0, NULL);
    readBackBuffer2->lpVtbl->Release(readBackBuffer2);

    // Verify the result
    bool equal = true;
    for (int i = 0; i < TEST_DATA_COUNT; i++)
    {
        if (resultBuffer[i] - 1 != s_dataBuffer0[i])
        {
            printf("%d index elements are not equal!\n", i);
            equal = false;
            break;
        }
    }
    if (equal) {
        puts("Verification 1 OK!");
    }
    printf("[0] = %d, [1] = %d, [2] = %d, [3] = %d\n", 
        resultBuffer2[0], resultBuffer2[1], resultBuffer2[2], resultBuffer2[3]);

    equal = true;
    for (int i = 4; i < TEST_DATA_COUNT; i++)
    {
        if (resultBuffer2[i] != s_dataBuffer1[i])
        {
            printf("%d index elements are not equal!\n", i);
            equal = false;
            break;
        }
    }
    if (equal) {
        puts("Verification 2 OK!");
    }

    free(resultBuffer);
    free(resultBuffer2);
}

// Release all the resources
void ReleaseResources(void)
{
    if (s_hEvent != NULL)
    {
        CloseHandle(s_hEvent);
        s_hEvent = NULL;
    }
    if (s_fence != NULL)
    {
        s_fence->lpVtbl->Release(s_fence);
        s_fence = NULL;
    }
    if (s_heap != NULL)
    {
        s_heap->lpVtbl->Release(s_heap);
        s_heap = NULL;
    }

    if (s_srcDataBuffer != NULL)
    {
        s_srcDataBuffer->lpVtbl->Release(s_srcDataBuffer);
        s_srcDataBuffer = NULL;
    }

    if (s_dstDataBuffer != NULL)
    {
        s_dstDataBuffer->lpVtbl->Release(s_dstDataBuffer);
        s_dstDataBuffer = NULL;
    }

    if (s_uploadBuffer != NULL)
    {
        s_uploadBuffer->lpVtbl->Release(s_uploadBuffer);
        s_uploadBuffer = NULL;
    }

    if (s_constantBuffer != NULL)
    {
        s_constantBuffer->lpVtbl->Release(s_constantBuffer);
        s_constantBuffer = NULL;
    }

    if (s_constantUploadBuffer != NULL)
    {
        s_constantUploadBuffer->lpVtbl->Release(s_constantUploadBuffer);
        s_constantUploadBuffer = NULL;
    }

    if (s_dst2Buffer != NULL)
    {
        s_dst2Buffer->lpVtbl->Release(s_dst2Buffer);
        s_dst2Buffer = NULL;
    }

    if (s_dst2UploadBuffer != NULL)
    {
        s_dst2UploadBuffer->lpVtbl->Release(s_dst2UploadBuffer);
        s_dst2UploadBuffer = NULL;
    }

    if (s_dataBuffer0 != NULL)
    {
        free(s_dataBuffer0);
        s_dataBuffer0 = NULL;
    }

    if (s_dataBuffer1 != NULL)
    {
        free(s_dataBuffer1);
        s_dataBuffer1 = NULL;
    }

    if (s_computeAllocator != NULL)
    {
        s_computeAllocator->lpVtbl->Release(s_computeAllocator);
        s_computeAllocator = NULL;
    }

    if (s_computeCommandList != NULL)
    {
        s_computeCommandList->lpVtbl->Release(s_computeCommandList);
        s_computeCommandList = NULL;
    }

    if (s_computeCommandQueue != NULL)
    {
        s_computeCommandQueue->lpVtbl->Release(s_computeCommandQueue);
        s_computeCommandQueue = NULL;
    }

    if (s_computeState != NULL)
    {
        s_computeState->lpVtbl->Release(s_computeState);
        s_computeState = NULL;
    }

    if (s_computeRootSignature != NULL)
    {
        s_computeRootSignature->lpVtbl->Release(s_computeRootSignature);
        s_computeRootSignature = NULL;
    }

    if (s_device != NULL)
    {
        s_device->lpVtbl->Release(s_device);
        s_device = NULL;
    }
    if (s_factory != NULL)
    {
        s_factory->lpVtbl->Release(s_factory);
        s_factory = NULL;
    }
}

int main(void)
{
    do
    {
        if (!CreateD3D12Device()) break;

        if (!CreateRootSignature()) break;

        if (!CreateComputePipelineStateObject()) break;

        if (!InitComputeCommands())
        {
            puts("InitComputeCommands failed!");
            break;
        }
        if (!CreateBuffers())
        {
            puts("CreateBuuffers failed!");
            break;
        }
        if (!CreateFenceAndEvent()) break;

        HRESULT hRes = s_computeCommandList->lpVtbl->Close(s_computeCommandList);
        if(FAILED(hRes))
        {
            fprintf(stderr, "Execute init commands failed: %ld\n", hRes);
            break;
        }

        s_computeCommandQueue->lpVtbl->ExecuteCommandLists(s_computeCommandQueue, 1, (ID3D12CommandList* const []) { (ID3D12CommandList*)s_computeCommandList });

        SyncCommandQueue(s_computeCommandQueue, s_device, COPY_SYNC_SIGNAL_VALUE);

        // After finishing the whole buffer copy operation,
        // the intermediate buffer s_uploadBuffer can be released now.
        if (s_uploadBuffer != NULL)
        {
            s_uploadBuffer->lpVtbl->Release(s_uploadBuffer);
            s_uploadBuffer = NULL;
        }
        if (s_constantUploadBuffer != NULL)
        {
            s_constantUploadBuffer->lpVtbl->Release(s_constantUploadBuffer);
            s_constantUploadBuffer = NULL;
        }
        if (s_dst2UploadBuffer != NULL)
        {
            s_dst2UploadBuffer->lpVtbl->Release(s_dst2UploadBuffer);
            s_dst2UploadBuffer = NULL;
        }

        DoCompute();
    }
    while (false);

    ReleaseResources();
}

