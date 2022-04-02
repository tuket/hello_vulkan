#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;
typedef const char* ConstStr;

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

GLFWwindow* window;

struct
{
    VkInstance inst;
    VkDevice device;
    VkSurfaceKHR surface;
} vk;

VkPhysicalDevice choosePhyisicalDevice()
{
    u32 numDevices;
    VkPhysicalDevice physicalDevices[16];
    vkEnumeratePhysicalDevices(vk.inst, &numDevices, physicalDevices);

    VkPhysicalDeviceProperties props[16];
    for(u32 i = 0; i < numDevices; i++)
        vkGetPhysicalDeviceProperties(physicalDevices[i], &props[i]);

    VkPhysicalDeviceMemoryProperties memProps[16];
    for(u32 i = 0; i < numDevices; i++)
        vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &memProps[i]);

    auto typeScore = [](VkPhysicalDeviceType t) -> u32 {
        switch(t) {
            case VK_PHYSICAL_DEVICE_TYPE_OTHER: return 1;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return 2;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 3;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 4;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 5;
        }
        return 0;
    };

    auto compare = [&](u32 i, u32 j) -> bool {
        const u32 inds[2] = {i, j};

        u32 typeScores[2];
        for(u32 i = 0; i < 2; i++)
            typeScores[i] = typeScore(props[inds[i]].deviceType);
        if(typeScores[0] != typeScores[1])
            return typeScores[0] < typeScores[1];
        
        u64 memSize[2] = {0};
        for(u32 i = 0; i < 2; i++) {
            const auto& memProp = memProps[i];
            memSize[i] += memProp.memoryHeaps[inds[i]].size;
        }
        return memSize[0] < memSize[1];
    };

    u32 bestDeviceInd = 0;
    for(u32 i = 1; i < numDevices; i++) {
        if(compare(bestDeviceInd, i))
            bestDeviceInd = i;
    }

    return physicalDevices[bestDeviceInd];
}

static void checkInstanceLayersAreSupported(int numLayers, ConstStr* layerNames)
{
    u32 numSupportedLayers;
    vkEnumerateInstanceLayerProperties(&numSupportedLayers, nullptr);
    if(numSupportedLayers == 0)
        return;
    auto supportedLayers = new VkLayerProperties[numSupportedLayers];
    vkEnumerateInstanceLayerProperties(&numSupportedLayers, supportedLayers);
    for(u32 i = 0; i < numLayers; i++) {
        u32 j;
        for(j = 0; j < numSupportedLayers; j++) {
            if(strcmp(layerNames[i], supportedLayers[j].layerName) == 0)
                break;
        }
        if(j == numSupportedLayers)
            printf("Layer '%s' is not supported\n", layerNames[i]);
    }
    delete[] supportedLayers;
}

static void checkDeviceExtensionSupported(VkPhysicalDevice device, u32 numExtensions, ConstStr* extensionNames)
{
    u32 numSupportedExtensions;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &numSupportedExtensions, nullptr);
    if(numSupportedExtensions == 0)
        return;
    auto supportedExtensions = new VkExtensionProperties[numSupportedExtensions];
    for(u32 i = 0; i < numExtensions; i++){
        u32 j;
        for(j = 0; j < numSupportedExtensions; j++) {
            if(strcmp(extensionNames[i], supportedExtensions[j].extensionName) == 0)
                break;
        }
        if(j == numSupportedExtensions)
            printf("Extension '%s' is not supported\n", extensionNames[i]);
    }
    delete[] supportedExtensions;
}

int main()
{
    { // create instance

        VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "hello2";
        appInfo.pEngineName = "none";
        appInfo.apiVersion = VK_API_VERSION_1_1;
        
        u32 numGlfwExtensions;
        ConstStr* glfwExtensions = glfwGetRequiredInstanceExtensions(&numGlfwExtensions);

        const char* layerNames[] = {
            "VK_LAYER_KHRONOS_validation"
        };
        const u32 numLayers = std::size(layerNames);
        checkInstanceLayersAreSupported(numLayers, layerNames);

        VkInstanceCreateInfo info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &appInfo;
        info.enabledExtensionCount = numGlfwExtensions;
        info.ppEnabledExtensionNames = glfwExtensions;
        info.enabledLayerCount = numLayers;
        info.ppEnabledLayerNames = layerNames;

        vkCreateInstance(&info, nullptr, &vk.inst);
    }

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // explicitly tell glfw not to create an OpengL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "hello2", nullptr, nullptr);
    glfwCreateWindowSurface(vk.inst, window, nullptr, &vk.surface);

    { // create device
        const VkPhysicalDevice physicalDevice = choosePhyisicalDevice();        
        ConstStr extensionNames[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        const u32 numExtensions = std::size(extensionNames);
        const auto physicalDevice = choosePhyisicalDevice();
        VkDeviceCreateInfo info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};

        u32 numFamilies;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> families(numFamilies);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numFamilies, &families[0]);
        u32 graphicsFamily, presentFamily;
        for(u32 i = 0; i < numFamilies; i++) {
            if(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                graphicsFamily = i;
            VkBool32 surfaceSupported;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, vk.surface, &surfaceSupported);
            if(surfaceSupported)
                presentFamily = i;
        }
        const float queuePriorities[1] = {0.f};
        u32 numQueues = 1;
        VkDeviceQueueCreateInfo queueInfos[2];
        queueInfos[0] = VkDeviceQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfos[0].queueCount = 1;
        queueInfos[0].queueFamilyIndex = graphicsFamily;
        if(graphicsFamily != presentFamily) {
            queueInfos[1] = VkDeviceQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueInfos[1].queueCount = 1;
            queueInfos[1].queueFamilyIndex = presentFamily;
            numQueues++;
        }

        info.enabledExtensionCount = numExtensions;
        info.ppEnabledExtensionNames = extensionNames;
        info.queueCreateInfoCount = numQueues;
        info.pQueueCreateInfos = queueInfos;
        vkCreateDevice(physicalDevice, &info, nullptr, &vk.device);

        delete[] families;
    }


}