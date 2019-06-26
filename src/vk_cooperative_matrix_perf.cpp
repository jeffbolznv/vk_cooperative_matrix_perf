/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <chrono>
#include <string.h>

#include <vulkan/vulkan.h>

using std::vector;

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))



#define CHECK_RESULT(r) do {    \
    if ((r) != VK_SUCCESS) {    \
        printf("result = %d, line = %d\n", (r), __LINE__);  \
        throw;  \
    }   \
} while (0)

// pasted from Vulkan spec
int32_t findProperties(const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
                       uint32_t memoryTypeBitsRequirement,
                       VkMemoryPropertyFlags requiredProperties) {
    const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
        const uint32_t memoryTypeBits = (1 << memoryIndex);
        const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

        const VkMemoryPropertyFlags properties =
            pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
        const bool hasRequiredProperties =
            (properties & requiredProperties) == requiredProperties;

        if (isRequiredMemoryType && hasRequiredProperties)
            return static_cast<int32_t>(memoryIndex);
    }

    // failed to find memory type
    return -1;
}

enum TestType
{
    TT_SHARED = 0,
    TT_TILED,
    TT_COUNT,
};


struct TestCase
{
    TestType testType;
    VkComponentTypeNV inputType;
    VkComponentTypeNV outputType;

    // MxNxK is the size of the full matrix multiply
    uint32_t M;
    uint32_t N;
    uint32_t K;

    // Each cooperative matrix multiply is lMxlNxlK
    uint32_t lM;
    uint32_t lN;
    uint32_t lK;

    // size of workgroup tile in destination matrix
    uint32_t TILE_M;
    uint32_t TILE_N;
    uint32_t TILE_K;

    bool BColMajor;
    uint32_t ARowLen;
    uint32_t ANumRows;
    uint32_t BRowLen;
    uint32_t BNumRows;
};

struct MatrixDesc
{
    struct
    {
        uint32_t rows, cols;
    } dims;
    VkComponentTypeNV dataType;
    size_t elementSize;
    VkDeviceSize bufferSize;
    uint32_t totalElements;

    // Create a host- and device-local buffer for each input and output.
    // Descriptors point at the device buffers.
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkDeviceMemory deviceMemory;
    void *ptr;

    void setData(uint32_t i, float value)
    {
        if (dataType == VK_COMPONENT_TYPE_FLOAT32_NV)
        {
            ((float *)ptr)[i] = value;
        }
        else
        {
            uint32_t asInt = *(uint32_t *)&value;
            int sign = (asInt & 0x80000000) >> 31;
            int exp = ((asInt & 0x7f800000) >> 23) - 127;
            int mantissa = (asInt & 0x7FFFFF);

            sign = sign << 15;
            exp = (exp + 15) << 10;
            mantissa = mantissa >> (23 - 10);

            if (asInt != 0) {
                asInt = sign | exp | mantissa;
            }

            ((uint16_t *)ptr)[i] = asInt;
        }
    }

    float getData(uint32_t i) const
    {
        if (dataType == VK_COMPONENT_TYPE_FLOAT32_NV)
        {
            return ((float *)ptr)[i];
        }
        else
        {
            uint32_t asInt = ((uint16_t *)ptr)[i];
            int sign = (asInt & 0x8000) >> 15;
            int exp = ((asInt & 0x7c00) >> 10) - 15;
            int mantissa = (asInt & 0x3FF);

            sign = sign << 31;
            exp = (exp + 127) << 23;
            mantissa = mantissa << (23 - 10);

            if (asInt != 0) {
                asInt = sign | exp | mantissa;
            }

            return *(float *)&asInt;
        }
    }

    float getData(int m, int n, bool colMajor) const
    {
        return getData(colMajor ? (n * dims.rows + m) : (m * dims.cols + n));
    }
};

// create storage for a matrix
void createMatrixDesc(VkDevice device, VkPhysicalDeviceMemoryProperties &memoryProperties, MatrixDesc &m, VkComponentTypeNV dt, int rows, int cols)
{
    VkResult result;

    m.dims.rows = rows;
    m.dims.cols = cols;
    m.dataType = dt;
    m.elementSize = m.dataType == VK_COMPONENT_TYPE_FLOAT16_NV ? 2 : 4;
    m.totalElements = m.dims.cols * m.dims.rows;
    m.bufferSize = m.totalElements * m.elementSize;

    VkBufferCreateInfo bufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        NULL,
        0,
        m.bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
        VK_SHARING_MODE_EXCLUSIVE,
        0u,
        NULL,
    };

    result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &m.hostBuffer);
    CHECK_RESULT(result);
    result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &m.deviceBuffer);
    CHECK_RESULT(result);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, m.hostBuffer, &memReqs);

    int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    int32_t deviceIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo memAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        NULL,
        memReqs.size,
        (uint32_t)hostIndex,
    };

    result = vkAllocateMemory(device, &memAllocateInfo, NULL, &m.hostMemory);
    CHECK_RESULT(result);

    memAllocateInfo.memoryTypeIndex = deviceIndex;
    result = vkAllocateMemory(device, &memAllocateInfo, NULL, &m.deviceMemory);
    CHECK_RESULT(result);

    result = vkBindBufferMemory(device, m.hostBuffer, m.hostMemory, 0);
    CHECK_RESULT(result);

    result = vkBindBufferMemory(device, m.deviceBuffer, m.deviceMemory, 0);
    CHECK_RESULT(result);

    result = vkMapMemory(device, m.hostMemory, 0, m.bufferSize, 0, &m.ptr);
    CHECK_RESULT(result);
}

// destroy storage for a matrix
void destroyMatrixDesc(VkDevice device, MatrixDesc &m)
{
    vkDestroyBuffer(device, m.hostBuffer, NULL);
    vkDestroyBuffer(device, m.deviceBuffer, NULL);
    vkFreeMemory(device, m.hostMemory, NULL);
    vkFreeMemory(device, m.deviceMemory, NULL);
}

int main(int argc, char *argv[])
{
    bool correctness = false;

    printf("usage: vk_cooperative_matrix_perf.exe [--correctness]\n\n");

    for (int arg = 1; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--correctness") == 0) {
            correctness = true;
        }
    }

    // Initialize Vulkan
    VkApplicationInfo applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        NULL,
        "Cooperative matrix performance test",
        1,
        "none",
        0,
        VK_MAKE_VERSION(1, 1, 0),
    };

    const char *enabledInstanceExtensions[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NULL,
        0,
        &applicationInfo,
        0,
        NULL,
        1,
        enabledInstanceExtensions,
    };

    VkResult result;
    VkInstance instance;
    result = vkCreateInstance(&instanceCreateInfo, NULL, &instance);
    CHECK_RESULT(result);

    uint32_t numPhysicalDevices = 0;
    vector<VkPhysicalDevice> physicalDevices;

    result = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, NULL);
    CHECK_RESULT(result);

    physicalDevices.resize(numPhysicalDevices);
    result = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, &physicalDevices[0]);
    CHECK_RESULT(result);

    int physicalDeviceIndex = -1;

    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {

        uint32_t numExtensions = 0;
        vector<VkExtensionProperties> extensions;

        result = vkEnumerateDeviceExtensionProperties(physicalDevices[i], NULL, &numExtensions, NULL);
        CHECK_RESULT(result);

        extensions.resize(numExtensions);
        result = vkEnumerateDeviceExtensionProperties(physicalDevices[i], NULL, &numExtensions, &extensions[0]);
        CHECK_RESULT(result);

        for (uint32_t j = 0; j < numExtensions; ++j) {
            if (strcmp(extensions[j].extensionName, VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
                physicalDeviceIndex = i;
                break;
            }
        }
        if (physicalDeviceIndex != -1) {
            break;
        }
    }

    if (physicalDeviceIndex == -1) {
        printf("couldn't find physical device that supports VK_NV_cooperative_matrix\n");
        return 0;
    }
    VkPhysicalDevice physicalDevice = physicalDevices[physicalDeviceIndex];


    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    uint32_t numQueueFamilies = 0;
    vector<VkQueueFamilyProperties> queueFamilies;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, NULL);

    queueFamilies.resize(numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, &queueFamilies[0]);

    int queueFamilyIndex = -1;

    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == -1) {
        printf("couldn't find compute queue\n");
        return 0;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        NULL,
        0,
        (uint32_t)queueFamilyIndex,
        1,
        &queuePriority,
    };

    // Query the list of supported cooperative matrix multiply sizes/types.
    uint32_t numCooperativeMatrixProperties = 0;
    vector<VkCooperativeMatrixPropertiesNV> cooperativeMatrixProperties;

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV");

    result = pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(physicalDevice, &numCooperativeMatrixProperties, NULL);
    CHECK_RESULT(result);

    cooperativeMatrixProperties.resize(numCooperativeMatrixProperties);
    for (uint32_t i = 0; i < numCooperativeMatrixProperties; ++i) {
        cooperativeMatrixProperties[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV;
        cooperativeMatrixProperties[i].pNext = NULL;
    }

    result = pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(physicalDevice, &numCooperativeMatrixProperties, &cooperativeMatrixProperties[0]);
    CHECK_RESULT(result);

    VkPhysicalDeviceCooperativeMatrixFeaturesNV coopMatFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
        NULL,
        VK_TRUE, // cooperativeMatrix
        VK_FALSE, // cooperativeMatrixRobustBufferAccess
    };

    VkPhysicalDeviceBufferAddressFeaturesEXT bufferDeviceAddressFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
        &coopMatFeatures,
        VK_TRUE, // bufferDeviceAddress
        VK_FALSE, // bufferDeviceAddressCaptureReplay
        VK_FALSE, // bufferDeviceAddressMultiDevice
    };

    VkPhysicalDeviceFloat16Int8FeaturesKHR float16Features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
        &bufferDeviceAddressFeatures,
        VK_TRUE, // shaderFloat16
        VK_FALSE, // shaderInt8
    };

    const char *enabledDeviceExtensions[] = { VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME,
                                              VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                                              VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME };
    VkDeviceCreateInfo deviceCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &float16Features,
        0,
        1,
        &deviceQueueCreateInfo,
        0,
        NULL,
        3,
        enabledDeviceExtensions,
        NULL,
    };

    VkDevice device;
    result = vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
    CHECK_RESULT(result);

    VkQueue queue;
    vkGetDeviceQueue(device, (uint32_t)queueFamilyIndex, 0, &queue);

    // The shaders use one UBO to pass in all the buffer addresses
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = 0;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        NULL,
        0,
        1,
        &layoutBinding,
    };

    VkDescriptorSetLayout descriptorSetLayout;
    result = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout);
    CHECK_RESULT(result);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        NULL,
        0,
        1,
        &descriptorSetLayout,
        0,
        NULL
    };

    VkPipelineLayout pipelineLayout;
    result = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout);
    CHECK_RESULT(result);

    VkDescriptorPoolSize poolSizes[1] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 } };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        NULL,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        1,
        ARRAY_LENGTH(poolSizes),
        poolSizes,
    };

    VkDescriptorPool descriptorPool;
    result = vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool);
    CHECK_RESULT(result);

    VkDescriptorSetAllocateInfo setAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        NULL,
        descriptorPool,
        1,
        &descriptorSetLayout,
    };

    VkDescriptorSet descriptorSet;
    result = vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet);
    CHECK_RESULT(result);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        NULL,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        (uint32_t)queueFamilyIndex,
    };

    VkCommandPool commandPool;
    result = vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
    CHECK_RESULT(result);

    // The command buffers, one for initializing buffers, one for compute, one
    // for reading back the results. This lets us time the compute work more
    // precisely.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        NULL,
        commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        3,
    };

    VkCommandBuffer commandBuffers[3];
    result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers);
    CHECK_RESULT(result);

    static const char *componentTypeString[] = {
        "fp16",
        "fp32",
        // ...
    };
    static const char *scopeString[] = {
        "invalid",
        "device",
        "workgroup",
        "subgroup",
        "invalid",
        "queuefamily",
    };

    // Loop over all shader types and all cooperative matrix properties.
    for (uint32_t tt = 0; tt < TT_COUNT; ++tt) {
    for (uint32_t i = 0; i < numCooperativeMatrixProperties; ++i) {

        VkCooperativeMatrixPropertiesNV *cooperativeMatrixProps = &cooperativeMatrixProperties[i];

        if (cooperativeMatrixProps->DType != VK_COMPONENT_TYPE_FLOAT16_NV &&
            cooperativeMatrixProps->DType != VK_COMPONENT_TYPE_FLOAT32_NV) {
            continue;
        }

        const char *fileName;
        switch (tt) {
        default:
            assert(0);
        case TT_SHARED:
            fileName = cooperativeMatrixProps->DType == VK_COMPONENT_TYPE_FLOAT16_NV ? "shaders/shmemfp16.spv" : "shaders/shmemfp32.spv";
            break;
        case TT_TILED:
            fileName = cooperativeMatrixProps->DType == VK_COMPONENT_TYPE_FLOAT16_NV ? "shaders/tiledfp16.spv" : "shaders/tiledfp32.spv";
            break;
        }

        printf("\nshader: %s\n", fileName);

        // Load and create the shader module.
        std::ifstream spirvfile(fileName, std::ios::binary | std::ios::ate);
        std::streampos spirvsize = spirvfile.tellg();
        if ((int)spirvsize == -1) {
            printf("%s not found!\n", fileName);
            throw;
        }
        spirvfile.seekg(0, std::ios::beg);

        vector<char> spirv(spirvsize);
        spirvfile.read(&spirv[0], spirvsize);

        VkShaderModuleCreateInfo shaderModuleCreateInfo = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            NULL,
            0,
            spirv.size(),
            (const uint32_t *)&spirv[0],
        };

        VkShaderModule shaderModule;
        result = vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule);
        CHECK_RESULT(result);


        printf("\ncooperativeMatrixProps = %dx%dx%d   A = %s B = %s C = %s D = %s scope = %s\n",
                cooperativeMatrixProps->MSize,
                cooperativeMatrixProps->NSize,
                cooperativeMatrixProps->KSize,
                componentTypeString[cooperativeMatrixProps->AType],
                componentTypeString[cooperativeMatrixProps->BType],
                componentTypeString[cooperativeMatrixProps->CType],
                componentTypeString[cooperativeMatrixProps->DType],
                scopeString[cooperativeMatrixProps->scope]);

        // For performance, test a 4096x4096x4096 multiply. For correctness,
        // test 256x256x256 (because the CPU reference computation is so slow).
        uint32_t defaultDim = correctness ? 256 : 4096;
        uint32_t defaultM = defaultDim;
        uint32_t defaultN = defaultDim;
        uint32_t defaultK = defaultDim;

        typedef struct {
            unsigned int maxTILE_M;
            unsigned int maxTILE_N;
            unsigned int granularityTILE_M;
            unsigned int granularityTILE_N;
        } SubTestParams;

        // TT_SHARED requires a multiple of 128x128 to satisfy the assumptions
        // of its SSBO->shared memory copy code.
        SubTestParams subTestParams[] = {
            { 256, 256, 128, 128 }, // TT_SHARED
            { 128, 128, cooperativeMatrixProps->MSize, cooperativeMatrixProps->NSize }, // TT_TILED
        };

        SubTestParams *params = &subTestParams[tt];

        for (unsigned int TILE_M_size = params->granularityTILE_M; TILE_M_size <= params->maxTILE_M; TILE_M_size += params->granularityTILE_M) {
        double maxPerfThisIter = 0;
        for (unsigned int TILE_N_size = params->granularityTILE_N; TILE_N_size <= params->maxTILE_N; TILE_N_size += params->granularityTILE_N) {
        for (unsigned int bcolmajor = 0; bcolmajor <= 1; ++bcolmajor) {

            bool BColMajor = bcolmajor != 0;
            TestCase testCase = {
                (TestType)tt, //TestType testType;
                cooperativeMatrixProps->AType, // VkComponentTypeNV inputType;
                cooperativeMatrixProps->DType, // VkComponentTypeNV outputType;

                // MxNxK is the size of the full matrix multiply
                defaultM, // uint32_t M;
                defaultN, // uint32_t N;
                defaultK, // uint32_t K;

                // Each cooperative matrix multiply is lMxlNxlK
                cooperativeMatrixProps->MSize, // uint32_t lM;
                cooperativeMatrixProps->NSize, // uint32_t lN;
                cooperativeMatrixProps->KSize, // uint32_t lK;

                // size of workgroup tile in destination matrix
                TILE_M_size, // uint32_t TILE_M;
                TILE_N_size, // uint32_t TILE_N;
                cooperativeMatrixProps->KSize, // uint32_t TILE_K;

                BColMajor, // bool BColMajor;
            };
            float alpha = 2.0f, beta = 3.0f;

            if (tt == TT_SHARED) {
                // These TILE_K sizes are what happens to perform better on current HW.
                if (cooperativeMatrixProps->DType == VK_COMPONENT_TYPE_FLOAT16_NV) {
                    testCase.TILE_K = 32;
                } else {
                    testCase.TILE_K = 16;
                }
                // This tile size is too slow and may TDR.
                if (cooperativeMatrixProps->DType == VK_COMPONENT_TYPE_FLOAT32_NV &&
                    testCase.TILE_M == 256 && testCase.TILE_N == 256) {
                    continue;
                }
            }

            // For non-power of two tile sizes, round up the matrix size to
            // be an even multiple of the tile size.
            testCase.M = (testCase.M + testCase.TILE_M - 1) / testCase.TILE_M * testCase.TILE_M;
            testCase.N = (testCase.N + testCase.TILE_N - 1) / testCase.TILE_N * testCase.TILE_N;
            testCase.K = (testCase.K + testCase.TILE_K - 1) / testCase.TILE_K * testCase.TILE_K;

            testCase.ARowLen = testCase.TILE_K;
            testCase.ANumRows = testCase.TILE_M;
            testCase.BRowLen = BColMajor ? testCase.TILE_K : testCase.TILE_N;
            testCase.BNumRows = BColMajor ? testCase.TILE_N : testCase.TILE_K;

            enum {MAT_A = 0, MAT_B = 1, MAT_C = 2, MAT_D = 3, NUM_MATS = 4};

            MatrixDesc matrices[NUM_MATS];

            createMatrixDesc(device, memoryProperties, matrices[MAT_A], cooperativeMatrixProps->AType, testCase.M, testCase.K);
            createMatrixDesc(device, memoryProperties, matrices[MAT_B], cooperativeMatrixProps->AType, testCase.K, testCase.N);
            createMatrixDesc(device, memoryProperties, matrices[MAT_C], cooperativeMatrixProps->DType, testCase.M, testCase.N);
            createMatrixDesc(device, memoryProperties, matrices[MAT_D], cooperativeMatrixProps->DType, testCase.M, testCase.N);

            // Allocate buffer to hold device addresses for the four matrices
            VkBuffer paramBuffer;
            VkDeviceMemory paramMemory;
            void *paramPtr;

            VkBufferCreateInfo bufferCreateInfo = {
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                NULL,
                0,
                4*sizeof(VkDeviceAddress),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_SHARING_MODE_EXCLUSIVE,
                0u,
                NULL,
            };

            result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &paramBuffer);
            CHECK_RESULT(result);

            VkMemoryRequirements memReqs;
            vkGetBufferMemoryRequirements(device, paramBuffer, &memReqs);

            int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

            VkMemoryAllocateInfo memAllocateInfo = {
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                NULL,
                memReqs.size,
                (uint32_t)hostIndex,
            };

            result = vkAllocateMemory(device, &memAllocateInfo, NULL, &paramMemory);
            CHECK_RESULT(result);

            result = vkBindBufferMemory(device, paramBuffer, paramMemory, 0);
            CHECK_RESULT(result);

            result = vkMapMemory(device, paramMemory, 0, bufferCreateInfo.size, 0, &paramPtr);
            CHECK_RESULT(result);

            PFN_vkGetBufferDeviceAddressEXT pfn_vkGetBufferDeviceAddressEXT =
                (PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressEXT");

            for (int i = 0; i < NUM_MATS; ++i) {
                MatrixDesc &m = matrices[i];

                VkBufferDeviceAddressInfoEXT info = {
                    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
                    NULL,
                    0,
                };
                VkDeviceAddress *addrsInMemory = (VkDeviceAddress *)paramPtr;
                info.buffer = m.deviceBuffer;
                VkDeviceAddress addr = pfn_vkGetBufferDeviceAddressEXT(device, &info);
                addrsInMemory[i] = addr;
            }

            VkDescriptorBufferInfo bufferDescriptor;
            bufferDescriptor.buffer = paramBuffer;
            bufferDescriptor.offset = 0;
            bufferDescriptor.range = bufferCreateInfo.size;

            VkWriteDescriptorSet writeDescriptorset = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,
                descriptorSet,
                0, // dstBinding,
                0, // dstArrayElement
                1, // descriptorCount
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                NULL,
                &bufferDescriptor,
                NULL,
            };

            vkUpdateDescriptorSets(device, 1, &writeDescriptorset, 0, NULL);

            // Initialize input buffers to random values. These are relatively
            // small and have few mantissa bits set so we don't lose precision
            // in fp16 mode when running the correctness test.
            // Initialize the output buffer to an obvious value.
            for (uint32_t i = 0; i < NUM_MATS; ++i) {
                MatrixDesc &m = matrices[i];
                for (uint32_t j = 0; j < m.totalElements; ++j) {
                    m.setData(j, ((float)(rand() & 0x3) - 1.0f) / 2.0f);
                    if (i == 3) {
                        m.setData(j, 1234.0f);
                    }
                }
            }

            // Specialize the shader with the matrix sizes, strides, and constants.
            const uint32_t specData[] = {
                testCase.lM,
                testCase.lN,
                testCase.lK,
                testCase.TILE_M,
                testCase.TILE_N,
                testCase.TILE_K,
                testCase.K,
                testCase.K, // stride0
                testCase.BColMajor ? testCase.K : testCase.N, // stride1
                testCase.N, // stride2
                testCase.N, // stride3
                *(uint32_t *)&alpha,
                *(uint32_t *)&beta,
                testCase.BColMajor,
                testCase.ARowLen,
                testCase.ANumRows,
                testCase.BRowLen,
                testCase.BNumRows,
            };

#if 0
            for (int i = 0; i < ARRAY_LENGTH(specData); ++i) {
                printf("specdata[%d] = %d\n", i, specData[i]);
            }
#endif

            VkSpecializationMapEntry entries[] = {
                {0, sizeof(uint32_t) * 0, sizeof(uint32_t)},
                {1, sizeof(uint32_t) * 1, sizeof(uint32_t)},
                {2, sizeof(uint32_t) * 2, sizeof(uint32_t)},
                {3, sizeof(uint32_t) * 3, sizeof(uint32_t)},
                {4, sizeof(uint32_t) * 4, sizeof(uint32_t)},
                {5, sizeof(uint32_t) * 5, sizeof(uint32_t)},
                {6, sizeof(uint32_t) * 6, sizeof(uint32_t)},
                {7, sizeof(uint32_t) * 7, sizeof(uint32_t)},
                {8, sizeof(uint32_t) * 8, sizeof(uint32_t)},
                {9, sizeof(uint32_t) * 9, sizeof(uint32_t)},
                {10, sizeof(uint32_t) * 10, sizeof(uint32_t)},
                {11, sizeof(uint32_t) * 11, sizeof(uint32_t)},
                {12, sizeof(uint32_t) * 12, sizeof(uint32_t)},
                {13, sizeof(uint32_t) * 13, sizeof(uint32_t)},
                {14, sizeof(uint32_t) * 14, sizeof(uint32_t)},
                {15, sizeof(uint32_t) * 15, sizeof(uint32_t)},
                {16, sizeof(uint32_t) * 16, sizeof(uint32_t)},
                {17, sizeof(uint32_t) * 17, sizeof(uint32_t)},
            };

            VkSpecializationInfo specInfo =
            {
                ARRAY_LENGTH(specData),
                entries,
                sizeof(specData),
                specData,
            };

            VkPipelineShaderStageCreateInfo shaderCreateInfo = {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                NULL,
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                shaderModule,
                "main",
                &specInfo,
            };

            VkComputePipelineCreateInfo pipelineCreateInfo = {
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                NULL,
                0,
                shaderCreateInfo,
                pipelineLayout,
                VK_NULL_HANDLE,
                0
            };

            VkPipeline pipeline;
            result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
            CHECK_RESULT(result);

            VkCommandBufferBeginInfo commandBufferBeginInfo = {
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                NULL,
                0,
                NULL,
            };

            // Download input buffers to device memory.
            result = vkBeginCommandBuffer(commandBuffers[0], &commandBufferBeginInfo);
            CHECK_RESULT(result);

            for (uint32_t i = 0; i < 4; ++i) {
                MatrixDesc &m = matrices[i];
                VkBufferCopy copy = { 0, 0, m.bufferSize };
                vkCmdCopyBuffer(commandBuffers[0], m.hostBuffer, m.deviceBuffer, 1, &copy);
            }

            result = vkEndCommandBuffer(commandBuffers[0]);
            CHECK_RESULT(result);

            VkSubmitInfo submitInfo = {
                VK_STRUCTURE_TYPE_SUBMIT_INFO,
                NULL,
                0,
                NULL,
                NULL,
                1,
                &commandBuffers[0],
                0,
                NULL,
            };

            submitInfo.pCommandBuffers = &commandBuffers[0];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            // Run the shader.
            result = vkBeginCommandBuffer(commandBuffers[1], &commandBufferBeginInfo);
            CHECK_RESULT(result);

            vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
            vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

            uint32_t repeatCount = correctness ? 1 : 10;

            for (uint32_t i = 0; i < repeatCount; ++i) {
                vkCmdDispatch(commandBuffers[1], testCase.N / testCase.TILE_N, testCase.M / testCase.TILE_M, 1);
            }

            result = vkEndCommandBuffer(commandBuffers[1]);
            CHECK_RESULT(result);

            if (!correctness) {
                // warmup submits, to get the clocks up before we run the timing
                submitInfo.pCommandBuffers = &commandBuffers[1];
                int warmupCount = tt == TT_SHARED ? 5 : 2;
                for (int i = 0; i < warmupCount; ++i) {
                    result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
                    CHECK_RESULT(result);
                    result = vkQueueWaitIdle(queue);
                    CHECK_RESULT(result);
                }
            }

            // Time the submit/wait time for this command buffer.
            auto beginTime = std::chrono::high_resolution_clock::now();

            submitInfo.pCommandBuffers = &commandBuffers[1];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            auto endTime = std::chrono::high_resolution_clock::now();
            uint64_t elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime).count();
            uint64_t flops = 2ULL * (uint64_t)testCase.M * (uint64_t)testCase.N * (uint64_t)testCase.K * (uint64_t)repeatCount;
            double tflops = (double)flops / (double)(elapsedUs / 1000000.0) / (1000.0*1000.0*1000.0*1000.0);

            printf("TILE_M=%d TILE_N=%d, TILE_K=%d BColMajor=%d ", testCase.TILE_M, testCase.TILE_N, testCase.TILE_K, testCase.BColMajor);
            if (!correctness) {
                printf("  %f TFlops\n", tflops);
            }

            // Upload the result from device memory.
            result = vkBeginCommandBuffer(commandBuffers[2], &commandBufferBeginInfo);
            CHECK_RESULT(result);
            {
                MatrixDesc &m = matrices[MAT_D];
                VkBufferCopy copy = { 0, 0, m.bufferSize };
                vkCmdCopyBuffer(commandBuffers[2], m.deviceBuffer, m.hostBuffer, 1, &copy);
            }
            result = vkEndCommandBuffer(commandBuffers[2]);
            CHECK_RESULT(result);

            submitInfo.pCommandBuffers = &commandBuffers[2];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            if (correctness)
            {
                const MatrixDesc &mat_a = matrices[MAT_A];
                const MatrixDesc &mat_b = matrices[MAT_B];
                const MatrixDesc &mat_c = matrices[MAT_C];
                const MatrixDesc &mat_d = matrices[MAT_D];
                bool pass = true;

                for (uint32_t i = 0; i < testCase.M; ++i)
                {
                    for (uint32_t j = 0; j < testCase.N; ++j)
                    {
                        float ref = 0;
                        for (uint32_t k = 0; k < testCase.K; ++k)
                        {
                            ref += mat_a.getData(i, k, false) * mat_b.getData(k, j, testCase.BColMajor);
                        }

                        ref = alpha*ref + beta*mat_c.getData(i, j, false);

                        float Dij = mat_d.getData(i, j, false);
                        if (ref != Dij) {
                            pass = false;
                            printf("error %d %d %f != %f\n", i, j, ref, Dij);
                        }
                    }
                }
                printf("%s\n", pass ? "pass" : "fail");
            }

            // Free the memory/buffers/pipeline for this iteration.
            for (int i = 0; i < NUM_MATS; ++i) {
                destroyMatrixDesc(device, matrices[i]);
            }
            vkDestroyPipeline(device, pipeline, NULL);

            if (maxPerfThisIter < tflops) {
                maxPerfThisIter = tflops;
            }
            // Stop this iteration (increasing tile size) if we've gotten to
            // the point where performance is decreasing. This usually means
            // the tile no longer fits in register file.
            if (!correctness && tflops < maxPerfThisIter / 2 && tt == TT_TILED) {
                break;
            }

        } // bcolmajor
        } // TILE_N_size
        } // TILE_M_size

        vkDestroyShaderModule(device, shaderModule, NULL);
    } // numCooperativeMatrixProperties
    } // TT_COUNT

    printf("\ndone\n");

    return 0;
}
