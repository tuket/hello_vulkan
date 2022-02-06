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

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef const char* const ConstStr;

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

#define SHADERS_PATH "shaders"
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

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    u32 graphicsQueueFamily, presentationQueueFamily;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> swapchainFramebuffers;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanup() {
        vkDestroySemaphore(device, renderFinishedSemaphores[0], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[0], nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        for (auto framebuffer : swapchainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapchainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapchain, nullptr);
        vkDestroyDevice(device, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "hello";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "hello_engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        static ConstStr layers[] = {
        #if !defined(NDEBUG)
            "VK_LAYER_KHRONOS_validation",
            //"VK_LAYER_LUNARG_api_dump",
        #endif
        };
        #if !defined(NDEBUG)
            u32 numSupportedLayers;
            vkEnumerateInstanceLayerProperties(&numSupportedLayers, nullptr);
            std::vector<VkLayerProperties> supportedLayers(numSupportedLayers);
            vkEnumerateInstanceLayerProperties(&numSupportedLayers, &supportedLayers[0]);
            for(ConstStr layer : layers) {
                bool supported = false;
                for(const auto& supportedLayer : supportedLayers) {
                    if(strcmp(supportedLayer.layerName, layer) == 0) {
                        supported = true;
                        break;
                    }
                }
                if(!supported) {
                    printf("Layer %s is not supported\n", layer);
                    //assert(false);
                }
            }
        #endif

        VkInstanceCreateInfo instInfo = {};
        instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo = &appInfo;
        instInfo.enabledLayerCount = std::size(layers);
        instInfo.ppEnabledLayerNames =  layers;
        u32 numGlfwExtensions;
        ConstStr* glfwExtensions = glfwGetRequiredInstanceExtensions(&numGlfwExtensions);
        std::vector<const char*> extensions;
        extensions.reserve(numGlfwExtensions + 1);
        for(u32 i = 0; i < numGlfwExtensions; i++)
            extensions.push_back(glfwExtensions[i]);
        #ifndef NDEBUG
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        #endif
        instInfo.enabledExtensionCount = extensions.size();
        instInfo.ppEnabledExtensionNames = extensions.data();
        
        if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
            printf("Error creating vulkan instance\n");
            exit(-1);
        }
    }

    void createSurface() {
        // create window surface
        if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            printf("Error: can't create window surface\n");
            exit(-1);
        }
    }

    void pickPhysicalDevice() {
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

    void createLogicalDevice() {
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
    }

    void createSwapChain() {
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
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // VK_SHARING_MODE_CONCURRENT
        //assert(presentationQueueFamily == vk.graphicsQueueFamily);
        //createInfo.queueFamilyIndexCount = 1;
        //createInfo.pQueueFamilyIndices = &vk.presentationQueueFamily;
        createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.clipped = VK_FALSE;
        //createInfo.oldSwapchain = ; // this can be used to recycle the old swapchain when resizing the window

        vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain);
    }

    void createImageViews() {
        u32 imageCount;
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        assert(imageCount == 2);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, &swapchainImages[0]);

        swapchainImageViews.resize(swapchainImages.size());
        printf("-------------------------------------------%d\n", int(swapchainImages.size()));

        for (size_t i = 0; i < swapchainImages.size(); i++) {
            VkImageViewCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = swapchainImages[i];
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

    void createRenderPass() {
        VkAttachmentDescription attachmentDesc = {};
        attachmentDesc.format = VK_FORMAT_B8G8R8A8_SRGB;
        attachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear before rendering
        attachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // store the result after rendering
        attachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we don't care. This doesn't guarantee that the contents of the image will be preserved, but that's not a problem since we are going to clear it anyways
        attachmentDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0; // the index in the attachemntDescs array (we only have one)
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // this is an output attachment so we use this enum for best performance

        VkSubpassDescription subpassDesc = {};
        subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.pColorAttachments = &colorAttachmentRef; // the index of the attachement in this array is referenced in the shader with "layout(location = 0) out vec4 o_color"
        
        // Subpasses in a render pass automatically take care of image layout transitions.
        // These transitions are controlled by subpass dependencies, which specify memory and execution dependencies between subpasses
        // We have only a single subpass right now, but the operations right before and right after this subpass also count as implicit "subpasses"
        // There are two built-in dependencies that take care of the transition at the start of the render pass and at the end of the render pass,
        // but the former does not occur at the right time. It assumes that the transition occurs at the start of the pipeline,
        // but we haven't acquired the image yet at that point!
        // There are two ways to deal with this problem:
        // 1) We could change the waitStages for the imageAvailableSemaphore to VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT to ensure that the
        //    render passes don't begin until the image is available
        // 2) We can make the render pass wait for the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT stage.
        // I've decided to go with the second option here, because it's a good excuse to have a look at subpass dependencies and how they work.
        VkSubpassDependency dependencies[2] = {};
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL; // refers to the implicit subpasses before and after the renderPass depending if it's "srcSubpass" or "dstSubpass"
        dependencies[0].dstSubpass = 0; // refers to our only subpass
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // specifies the stage of the pipeline after blending where the final color values are output from the pipeline;
        dependencies[0].srcAccessMask = 0; // TODO: I don't understand this: https://github.com/ARM-software/vulkan-sdk/issues/14
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0; // refers to the implicit subpasses before and after the renderPass depending if it's "srcSubpass" or "dstSubpass"
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL; // refers to our only subpass
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // specifies the stage of the pipeline after blending where the final color values are output from the pipeline;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // TODO: I don't understand this: https://github.com/ARM-software/vulkan-sdk/issues/14
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; // Do not block any subsequent work
        dependencies[1].dstAccessMask = 0;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &attachmentDesc;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDesc;
        //renderPassInfo.dependencyCount = 1;
        //renderPassInfo.pDependencies = &dependency;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = dependencies;

        if(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            printf("Error creating the renderPass\n");
            exit(-1);
        }
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

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stagesInfos;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShadModule, nullptr);
        vkDestroyShaderModule(device, vertShadModule, nullptr);
    }

    void createFramebuffers() {
        swapchainFramebuffers.resize(swapchainImageViews.size());

        for (size_t i = 0; i < swapchainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapchainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = WIDTH;
            framebufferInfo.height = HEIGHT;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = graphicsQueueFamily;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(swapchainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

            if(vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                printf("Error beginning cmd buffer\n");
                exit(-1);
            }

            VkRenderPassBeginInfo rpBeginInfo = {};
            rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpBeginInfo.renderPass = renderPass;
            rpBeginInfo.framebuffer = swapchainFramebuffers[i];
            rpBeginInfo.renderArea = {{0,0}, {WIDTH, HEIGHT}};
            const VkClearValue CLEAR_VALUE = {};
            rpBeginInfo.clearValueCount = 1;
            rpBeginInfo.pClearValues = &CLEAR_VALUE;

            vkCmdBeginRenderPass(commandBuffers[i], &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            // VK_SUBPASS_CONTENTS_INLINE specifies that the contents of the subpass will be recorded inline in the
            //      primary command buffer, and secondary command buffers must not be executed within the subpass.
            // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS specifies that the contents are recorded in secondary
            //      command buffers that will be called from the primary command buffer, and vkCmdExecuteCommands
            //      is the only valid command on the command buffer until vkCmdNextSubpass or vkCmdEndRenderPass.

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // finally draw the triangle!
            vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);

            if(vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                printf("Error ending cmd buffer\n");
                exit(-1);
            }
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(1);
        renderFinishedSemaphores.resize(1);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        for (size_t i = 0; i < 1; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void drawFrame() {
        u32 imgIndex;
        vkAcquireNextImageKHR(device,
            swapchain,
            1'000'000'000, // timeout in nanoseconds
            imageAvailableSemaphores[0],
            VK_NULL_HANDLE, // fence
            &imgIndex); // we will use this index for addressing the corresponding cmd buffer

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphores[0];
        const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imgIndex];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphores[0];

        //vkQueueWaitIdle(vk.graphicsQueue);
        if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            printf("Error submitting cmd buffers");
            exit(-1);
        }

        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores[0];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imgIndex;

        vkQueuePresentKHR(presentQueue, &presentInfo);
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