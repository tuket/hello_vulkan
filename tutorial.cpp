#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>
#include <assert.h>

#define SHADERS_PATH "shaders"

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef const char* const ConstStr;

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {
public:
    void run() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

        initVulkan();
        //mainLoop();
    }

private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    u32 graphicsQueueFamily;
    u32 presentationQueueFamily;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapchain;
    VkImage swapchainImages[2];
    //VkFormat swapchainImageFormat;
    //VkExtent2D swapchainExtent;
    VkImageView swapchainImageViews[2];

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    void initVulkan() {
        { // create vulkan instance
            if (enableValidationLayers && !checkValidationLayerSupport()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }

            VkApplicationInfo appInfo{};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Hello Triangle";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;

            VkInstanceCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;

            auto extensions = getRequiredExtensions();
            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();

            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
                throw std::runtime_error("failed to create instance!");
            }
        }

        // create window surface
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        { // pick physical device

            u32 numPhysicalDevices;
            vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, nullptr);
            std::vector<VkPhysicalDevice> physicalDevices(numPhysicalDevices);
            if(numPhysicalDevices == 0) {
                printf("Error: there are no devices supporting Vulkan\n");
                exit(-1);
            }
            vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, &physicalDevices[0]);

            auto compareProps = [](
                const VkPhysicalDeviceProperties& propsA,
                const VkPhysicalDeviceMemoryProperties& memPropsA,
                const VkPhysicalDeviceProperties& propsB,
                const VkPhysicalDeviceMemoryProperties& memPropsB) -> bool
            {
                auto calcDeviceTypeScore = [](VkPhysicalDeviceType a) -> u8 {
                    switch(a) {
                        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
                        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
                        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
                        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
                        default: return 0;
                    }
                };
                const u8 scoreA = calcDeviceTypeScore(propsA.deviceType);
                const u8 scoreB = calcDeviceTypeScore(propsB.deviceType);
                if(scoreA != scoreB)
                    return scoreA < scoreB;

                auto calcMem = [](const VkPhysicalDeviceMemoryProperties& a) -> u64
                {
                    u64 mem = 0;
                    for(u32 i = 0; i < a.memoryHeapCount; i++) {
                        if(a.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                            mem = std::max(mem, a.memoryHeaps[i].size);
                    }
                    return mem;
                };
                u32 memA = calcMem(memPropsA);
                u32 memB = calcMem(memPropsB);

                return memA < memB;
            };

            VkPhysicalDeviceProperties bestProps;
            vkGetPhysicalDeviceProperties(physicalDevices[0], &bestProps);
            VkPhysicalDeviceMemoryProperties bestMemProps;
            vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &bestMemProps);
            u32 bestI = 0;
            for(u32 i = 1; i < numPhysicalDevices; i++) {
                VkPhysicalDeviceProperties props;
                VkPhysicalDeviceMemoryProperties memProps;
                vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
                vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &memProps);
                if(compareProps(bestProps, bestMemProps, props, memProps)) {
                    bestProps = props;
                    bestMemProps = memProps;
                    bestI = i;
                }
            }
            physicalDevice = physicalDevices[bestI];
        }

        { // create logical device
            u32 numQueueFamilies;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilyProps(numQueueFamilies);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, &queueFamilyProps[0]);

            graphicsQueueFamily = numQueueFamilies;
            presentationQueueFamily = numQueueFamilies;
            for(u32 i = 0; i < numQueueFamilies; i++) {
                if(queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                    graphicsQueueFamily = i;
                VkBool32 supportPresentation;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supportPresentation);
                if(supportPresentation)
                    presentationQueueFamily = i;
            }

            if(graphicsQueueFamily == numQueueFamilies) {
                printf("Error: there is no queue that supports graphics\n");
                exit(-1);
            }
            if(presentationQueueFamily == numQueueFamilies) {
                printf("Error: there is no queue that supports presentation\n");
                exit(-1);
            }

            u32 queueFamilyInds[2] = {graphicsQueueFamily};
            u32 numQueues;
            if(graphicsQueueFamily == presentationQueueFamily) {
                numQueues = 1;
            }
            else {
                numQueues = 2;
                queueFamilyInds[1] = graphicsQueueFamily;
            }

            const float queuePriorities[] = {1.f};
            VkDeviceQueueCreateInfo queueCreateInfos[2] = {};
            for(u32 i = 0; i < numQueues; i++)
            {
                queueCreateInfos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfos[i].queueFamilyIndex = queueFamilyInds[i];
                queueCreateInfos[i].queueCount = 1;
                queueCreateInfos[i].pQueuePriorities = queuePriorities;
            }

            VkDeviceCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            info.queueCreateInfoCount = numQueues;
            info.pQueueCreateInfos = queueCreateInfos;
            info.enabledLayerCount = 0;
            info.ppEnabledLayerNames = nullptr;
            ConstStr deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
            info.enabledExtensionCount = std::size(deviceExtensions);
            info.ppEnabledExtensionNames = deviceExtensions;
            const VkResult deviceCreatedOk = vkCreateDevice(physicalDevice, &info, nullptr, &device);
            if(deviceCreatedOk != VK_SUCCESS) {
                printf("Error: couldn't create device\n");
                exit(-1);
            }

            vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
            vkGetDeviceQueue(device, presentationQueueFamily, 0, &presentQueue);

            // queues
            // https://community.khronos.org/t/guidelines-for-selecting-queues-and-families/7222
            // https://www.reddit.com/r/vulkan/comments/aara8f/best_way_for_selecting_queuefamilies/
            // https://stackoverflow.com/questions/37575012/should-i-try-to-use-as-many-queues-as-possible
        }

            { // crete swapchain
        VkSurfaceCapabilitiesKHR surfaceCpabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCpabilities);

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = 2;
        createInfo.imageFormat = VK_FORMAT_B8G8R8A8_SRGB; // TODO: I think this format has mandatory support but I'm not sure
        createInfo.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        createInfo.imageExtent = surfaceCpabilities.currentExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 1;
        createInfo.pQueueFamilyIndices = &presentationQueueFamily;
        createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.clipped = VK_FALSE;
        //createInfo.oldSwapchain = ; // this can be used to recycle the old swapchain when resizing the window

        vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain);
    }

    { // create image views of the swapchain
        u32 imageCount;
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        assert(imageCount == 2);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages);

        for(u32 i = 0; i < 2; i++)
        {
            VkImageViewCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = swapchainImages[0];
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = VK_FORMAT_B8G8R8A8_SRGB;
            // info.components = ; // channel swizzling VK_COMPONENT_SWIZZLE_IDENTITY is 0
            info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            info.subresourceRange.baseMipLevel = 0;
            info.subresourceRange.levelCount = 1;
            info.subresourceRange.baseArrayLayer = 0;
            info.subresourceRange.layerCount = 1;

            vkCreateImageView(device, &info, nullptr, &swapchainImageViews[i]);
        }
    }

        createGraphicsPipeline();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    VkPipelineShaderStageCreateInfo makeStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule module)
    {
        VkPipelineShaderStageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = module;
        info.pName = "main";
        info.pSpecializationInfo = nullptr; // allows to specify values for shader constants
        return info;
    }

    void createGraphicsPipeline() {
        VkShaderModule vertShadModule = createShaderModule(SHADERS_PATH"/simple.vert.glsl.spv");
        VkShaderModule fragShadModule = createShaderModule(SHADERS_PATH"/simple.frag.glsl.spv");

        const VkPipelineShaderStageCreateInfo stagesInfos[] = {
            makeStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShadModule),
            makeStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShadModule)
        };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) WIDTH;
        viewport.height = (float) HEIGHT;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {WIDTH, HEIGHT};

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizerInfo = {};
        rasterizerInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizerInfo.depthClampEnable = VK_FALSE; // enabling will clamp depth instead of discarding which can be useful when rendering shadowmaps
        rasterizerInfo.rasterizerDiscardEnable = VK_FALSE; // if enable discards all geometry (could be useful for transformfeedback)
        rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizerInfo.lineWidth = 1.f;
        rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizerInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizerInfo.depthBiasEnable = VK_FALSE; // useful for shadow mapping


        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_G_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo colorBlendInfo = {};
        colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        //colorBlendInfo.logicOpEnable = VK_FALSE;
        //colorBlendInfo.logicOp = VK_LOGIC_OP_COPY;
        colorBlendInfo.attachmentCount = 1;
        colorBlendInfo.pAttachments = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkPipelineLayout pipelineLayout;
        const auto pipelineLayoutRet = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (pipelineLayoutRet != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        vkDestroyShaderModule(device, vertShadModule, nullptr);
        vkDestroyShaderModule(device, fragShadModule, nullptr);

        // -- create the renderPass ---
        VkAttachmentDescription attachmentDesc = {};
        attachmentDesc.format = VK_FORMAT_B8G8R8A8_SRGB;
        attachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear before rendering
        attachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // store the result after rendering
        attachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we don't care. This doesn't guarantee that the contents of th eimage will be preserved, but that's not a problem since we are going to clear it anyways
        attachmentDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0; // the index in the attachemntDescs array (we only have one)
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // this is an output attachment so we use this enum for best performance

        VkSubpassDescription subpassDesc = {};
        subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.pColorAttachments = &colorAttachmentRef; // the index of the attachement in this array is referenced in the shader with "layout(location = 0) out vec4 o_color"

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &attachmentDesc;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDesc;

        if(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            printf("Error creating the renderPass\n");
            exit(-1);
        }

        // -- finally create the pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stagesInfos;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizerInfo;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlendInfo;
        //pipelineInfo.pDynamicState = &dynamicStateInfo;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass; // render pass describing the enviroment in which the pipeline will be used
            // the pipeline must only be used with a render pass compatilble with this one
        pipelineInfo.subpass = 0; // index of the subpass in the render pass where this pipeline will be used
        //pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // you can derive from another pipeline
        //pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShadModule, nullptr);
        vkDestroyShaderModule(device, vertShadModule, nullptr);
    }

    char* readBinFile(int& len, const char* fileName)
    {
        FILE* file = fopen(fileName, "rb");
        if(!file)
            return nullptr;
        fseek(file, 0, SEEK_END);
        len = ftell(file);
        rewind(file);
        char* txt = new char[len];
        fread(txt, len, 1, file);
        fclose(file);
        return txt;
    }

    VkShaderModule createShaderModule(const char* fileName)
    {
        int len;
        char* data = readBinFile(len, fileName);
        if(!data) {
            printf("Error loading shader spir-v\n");
            exit(-1);
        }
        VkShaderModuleCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = len;
        info.pCode = (u32*)data;
        VkShaderModule module;
        const VkResult res = vkCreateShaderModule(device, &info, nullptr, &module);
        if(res != VK_SUCCESS) {
            printf("Error: could not create vertex shader module\n");
            exit(-1);
        }
        delete[] data;
        return module;
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
