#include <stdio.h>
#include <string.h>
#include <vector>
#include <utility>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <time.h>

typedef const char* const ConstStr;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

GLFWwindow* window;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

struct VulkanData
{
    VkInstance inst;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    u32 graphicsQueueFamily, presentationQueueFamily;
    VkQueue graphicsQueue;
    VkQueue presentationQueue;
    VkSwapchainKHR swapchain;
    VkImageView swapchainImageViews[2];
    VkFramebuffer framebuffers[2];
    VkRenderPass renderPass;
    VkPipeline pipeline;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuffers[2];
    VkSemaphore semaphore_swapchainImgAvailable[2];
    VkSemaphore semaphore_drawFinished[2];
    VkFence fence_queueWorkFinished[2];
    u32 frameInd = 0;
} vk;

static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    printf("%s\n", pCallbackData->pMessage);

    return VK_FALSE;
}

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
    const VkResult res = vkCreateShaderModule(vk.device, &info, nullptr, &module);
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

void initVulkan()
{
    { // create vulkan instance
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "hello";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "hello_engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        #if defined(NDEBUG)
            static ConstStr* layers = nullptr;
            constexpr size_t numLayers = 0;
            //static ConstStr layers[] = {"VK_LAYER_MESA_overlay"};
            //constexpr size_t numLayers = 1;
        #else
            static ConstStr layers[] = {
                "VK_LAYER_KHRONOS_validation",
                //"VK_LAYER_LUNARG_api_dump",
            };
            constexpr size_t numLayers = std::size(layers);

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
        instInfo.enabledLayerCount = numLayers;
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
        
        if (vkCreateInstance(&instInfo, nullptr, &vk.inst) != VK_SUCCESS) {
            printf("Error creating vulkan instance\n");
            exit(-1);
        }
    }

    // create window surface
    if(glfwCreateWindowSurface(vk.inst, window, nullptr, &vk.surface) != VK_SUCCESS) {
        printf("Error: can't create window surface\n");
        exit(-1);
    }

    { // pick physical device

        u32 numPhysicalDevices;
        vkEnumeratePhysicalDevices(vk.inst, &numPhysicalDevices, nullptr);
        std::vector<VkPhysicalDevice> physicalDevices(numPhysicalDevices);
        if(numPhysicalDevices == 0) {
            printf("Error: there are no devices supporting Vulkan\n");
            exit(-1);
        }
        vkEnumeratePhysicalDevices(vk.inst, &numPhysicalDevices, &physicalDevices[0]);

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
        vk.physicalDevice = physicalDevices[bestI];
    }

    { // create logical device
        u32 numQueueFamilies;
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &numQueueFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProps(numQueueFamilies);
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &numQueueFamilies, &queueFamilyProps[0]);
        u32 graphicsFamilyInd = -1, presentationFamilyInd = -1;
        for(u32 i = 0; i < numQueueFamilies; i++) {
            const bool supportsGraphics = queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
            VkQueueFamilySwa
        }

        vk.graphicsQueueFamily = numQueueFamilies;
        vk.presentationQueueFamily = numQueueFamilies;
        for(u32 i = 0; i < numQueueFamilies; i++) {
            if(queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                vk.graphicsQueueFamily = i;
            VkBool32 supportPresentation;
            vkGetPhysicalDeviceSurfaceSupportKHR(vk.physicalDevice, i, vk.surface, &supportPresentation);
            if(supportPresentation)
                vk.presentationQueueFamily = i;
        }

        if(vk.graphicsQueueFamily == numQueueFamilies) {
            printf("Error: there is no queue that supports graphics\n");
            exit(-1);
        }
        if(vk.presentationQueueFamily == numQueueFamilies) {
            printf("Error: there is no queue that supports presentation\n");
            exit(-1);
        }

        u32 queueFamilyInds[2] = {vk.graphicsQueueFamily};
        u32 numQueues;
        if(vk.graphicsQueueFamily == vk.presentationQueueFamily) {
            numQueues = 1;
        }
        else {
            numQueues = 2;
            queueFamilyInds[1] = vk.graphicsQueueFamily;
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
        const VkResult deviceCreatedOk = vkCreateDevice(vk.physicalDevice, &info, nullptr, &vk.device);
        if(deviceCreatedOk != VK_SUCCESS) {
            printf("Error: couldn't create device\n");
            exit(-1);
        }

        vkGetDeviceQueue(vk.device, vk.graphicsQueueFamily, 0, &vk.graphicsQueue);
        vkGetDeviceQueue(vk.device, vk.presentationQueueFamily, 0, &vk.presentationQueue);

        // queues
        // https://community.khronos.org/t/guidelines-for-selecting-queues-and-families/7222
        // https://www.reddit.com/r/vulkan/comments/aara8f/best_way_for_selecting_queuefamilies/
        // https://stackoverflow.com/questions/37575012/should-i-try-to-use-as-many-queues-as-possible
    }

    { // create swapchain
        VkSurfaceCapabilitiesKHR surfaceCpabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physicalDevice, vk.surface, &surfaceCpabilities);

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = vk.surface;
        createInfo.minImageCount = 2;
        createInfo.imageFormat = VK_FORMAT_B8G8R8A8_SRGB; // TODO: I think this format has mandatory support but I'm not sure
        createInfo.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        createInfo.imageExtent = surfaceCpabilities.currentExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // VK_SHARING_MODE_CONCURRENT
        assert(vk.presentationQueueFamily == vk.graphicsQueueFamily);
        //createInfo.queueFamilyIndexCount = 1;
        //createInfo.pQueueFamilyIndices = &vk.presentationQueueFamily;
        createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        //createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        createInfo.clipped = VK_FALSE;
        //createInfo.oldSwapchain = ; // this can be used to recycle the old swapchain when resizing the window

        vkCreateSwapchainKHR(vk.device, &createInfo, nullptr, &vk.swapchain);
    }

    { // create image views of the swapchain
        u32 imageCount;
        VkImage images[2];
        vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &imageCount, nullptr);
        assert(imageCount == 2);
        vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &imageCount, images);

        for(u32 i = 0; i < 2; i++)
        {
            VkImageViewCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = images[i];
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = VK_FORMAT_B8G8R8A8_SRGB;
            // info.components = ; // channel swizzling VK_COMPONENT_SWIZZLE_IDENTITY is 0
            info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            info.subresourceRange.baseMipLevel = 0;
            info.subresourceRange.levelCount = 1;
            info.subresourceRange.baseArrayLayer = 0;
            info.subresourceRange.layerCount = 1;

            vkCreateImageView(vk.device, &info, nullptr, &vk.swapchainImageViews[i]);
        }
    }

    { // create graphics pipeline
        VkShaderModule vertShadModule = createShaderModule(SHADERS_PATH"/simple.vert.glsl.spv");
        VkShaderModule fragShadModule = createShaderModule(SHADERS_PATH"/simple.frag.glsl.spv");

        const VkPipelineShaderStageCreateInfo stagesInfos[] = {
            makeStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShadModule),
            makeStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShadModule)
        };

        VkPipelineVertexInputStateCreateInfo vertInputInfo = {};
        vertInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertInputInfo.vertexBindingDescriptionCount = 0;
        vertInputInfo.pVertexBindingDescriptions = nullptr;
        vertInputInfo.vertexAttributeDescriptionCount = 0;
        vertInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo assemblyInfo = {};
        assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        assemblyInfo.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport;
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = WINDOW_WIDTH;
        viewport.height = WINDOW_HEIGHT;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;
        VkRect2D scissor;
        scissor.offset = {0, 0};
        scissor.extent = {WINDOW_WIDTH, WINDOW_HEIGHT};
        VkPipelineViewportStateCreateInfo viewportInfo = {};
        viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportInfo.viewportCount = 1;
        viewportInfo.pViewports = &viewport;
        viewportInfo.scissorCount = 1;
        viewportInfo.pScissors = &scissor;
        
        VkPipelineRasterizationStateCreateInfo rasterizerInfo = {};
        rasterizerInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizerInfo.depthClampEnable = VK_FALSE; // enabling will clamp depth instead of discarding which can be useful when rendering shadowmaps
        rasterizerInfo.rasterizerDiscardEnable = VK_FALSE; // if enable discards all geometry (could be useful for transformfeedback)
        rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizerInfo.lineWidth = 1.f;
        rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizerInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizerInfo.depthBiasEnable = VK_FALSE; // useful for shadow mapping

        VkPipelineMultisampleStateCreateInfo multisampleInfo = {}; // multisampling: it works by combining the fragment shader results of multiple polygons that rasterize to the same pixel
        multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleInfo.sampleShadingEnable = VK_FALSE;
        multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        //VkPipelineDepthStencilStateCreateInfo depthStencilInfo = {};

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo colorBlendInfo = {};
        colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        //colorBlendInfo.logicOpEnable = VK_FALSE;
        //colorBlendInfo.logicOp = VK_LOGIC_OP_COPY;
        colorBlendInfo.attachmentCount = 1;
        colorBlendInfo.pAttachments = &colorBlendAttachment;

        //const VkDynamicState dynamicStates[] = {
        //    VK_DYNAMIC_STATE_VIEWPORT,
        //    VK_DYNAMIC_STATE_SCISSOR
        //};
        //VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
        //dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        //dynamicStateInfo.dynamicStateCount = 2;
        //dynamicStateInfo.pDynamicStates = dynamicStates;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkPipelineLayout pipelineLayout;
        vkCreatePipelineLayout(vk.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        // -- create the renderPass ---
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

        if(vkCreateRenderPass(vk.device, &renderPassInfo, nullptr, &vk.renderPass) != VK_SUCCESS) {
            printf("Error creating the renderPass\n");
            exit(-1);
        }

        // -- finally create the pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stagesInfos;
        pipelineInfo.pVertexInputState = &vertInputInfo;
        pipelineInfo.pInputAssemblyState = &assemblyInfo;
        pipelineInfo.pViewportState = &viewportInfo;
        pipelineInfo.pRasterizationState = &rasterizerInfo;
        pipelineInfo.pMultisampleState = &multisampleInfo;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlendInfo;
        //pipelineInfo.pDynamicState = &dynamicStateInfo;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = vk.renderPass; // render pass describing the enviroment in which the pipeline will be used
            // the pipeline must only be used with a render pass compatible with this one
        pipelineInfo.subpass = 0; // index of the subpass in the render pass where this pipeline will be used
        //pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // you can derive from another pipeline
        //pipelineInfo.basePipelineIndex = -1;

        if(vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk.pipeline) != VK_SUCCESS) {
            printf("Error creating the graphics piepeline\n");
            exit(-1);
        }

        vkDestroyShaderModule(vk.device, fragShadModule, nullptr);
        vkDestroyShaderModule(vk.device, vertShadModule, nullptr);
    }

    // create framebuffers
    for(int i = 0; i < 2; i++)
    {
        VkFramebufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass = vk.renderPass; // reference renderPass defines compatibility. We will be able to use this framebuffer only with compatible renderPasses
        info.attachmentCount = 1;
        info.pAttachments = &vk.swapchainImageViews[i];
        info.width = WINDOW_WIDTH;
        info.height = WINDOW_HEIGHT;
        info.layers = 1; // used for image arrays

        const auto cmdRes = vkCreateFramebuffer(vk.device, &info, nullptr, &vk.framebuffers[i]);
        if(cmdRes != VK_SUCCESS) {
            printf("Erro creating the framebuffer\n");
            exit(-1);
        }
    }

    // create cmd pool
    {
        VkCommandPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.queueFamilyIndex = vk.graphicsQueueFamily;
        if(vkCreateCommandPool(vk.device, &info, nullptr, &vk.cmdPool) != VK_SUCCESS) {
            printf("Error creating cmd pool\n");
            exit(-1);
        }
    }

    { // create cmd buffer
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = vk.cmdPool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // secondary cmdBuffer can be called from primary cmdBuffers
        info.commandBufferCount = 2; // we have 2 cmd buffers so we can record the next frame while the current is still drawing

        if(vkAllocateCommandBuffers(vk.device, &info, vk.cmdBuffers) != VK_SUCCESS) {
            printf("Error creating cmd buffers\n");
            exit(-1);
        }
    }

    // record cmd buffers
    for(int i = 0; i < 2; i++)
    {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if(vkBeginCommandBuffer(vk.cmdBuffers[i], &beginInfo) != VK_SUCCESS) {
            printf("Error beginning cmd buffer\n");
            exit(-1);
        }

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = vk.renderPass;
        rpBeginInfo.framebuffer = vk.framebuffers[i];
        rpBeginInfo.renderArea = {{0,0}, {WINDOW_WIDTH, WINDOW_HEIGHT}};
        const VkClearValue CLEAR_VALUE = {};
        rpBeginInfo.clearValueCount = 1;
        rpBeginInfo.pClearValues = &CLEAR_VALUE;

        vkCmdBeginRenderPass(vk.cmdBuffers[i], &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        // VK_SUBPASS_CONTENTS_INLINE specifies that the contents of the subpass will be recorded inline in the
        //      primary command buffer, and secondary command buffers must not be executed within the subpass.
        // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS specifies that the contents are recorded in secondary
        //      command buffers that will be called from the primary command buffer, and vkCmdExecuteCommands
        //      is the only valid command on the command buffer until vkCmdNextSubpass or vkCmdEndRenderPass.

        vkCmdBindPipeline(vk.cmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);

        // finally draw the triangle!
        vkCmdDraw(vk.cmdBuffers[i], 3, 1, 0, 0);

        vkCmdEndRenderPass(vk.cmdBuffers[i]);

        if(vkEndCommandBuffer(vk.cmdBuffers[i]) != VK_SUCCESS) {
            printf("Error ending cmd buffer\n");
            exit(-1);
        }
    }

    // create semaphores for synchonizing the swapchain presentation
    {
        VkSemaphoreCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkSemaphore semaphores[4];
        for(int i = 0; i < 4; i++) {
            if(vkCreateSemaphore(vk.device, &info, nullptr, &semaphores[i]) != VK_SUCCESS) {
                printf("Error creating semaphores\n");
                exit(-1);
            }
        }
        vk.semaphore_swapchainImgAvailable[0] = semaphores[0];
        vk.semaphore_swapchainImgAvailable[1] = semaphores[1];
        vk.semaphore_drawFinished[0] = semaphores[2];
        vk.semaphore_drawFinished[1] = semaphores[3];
    }

    { // create fences for synchonizing swapchain synchronization
        VkFenceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for(int i = 0; i < 2; i++)
            vkCreateFence(vk.device, &info, nullptr, &vk.fence_queueWorkFinished[i]);
    }
}

void draw()
{
    u32 imgIndex;
    vkAcquireNextImageKHR(vk.device,
        vk.swapchain,
        1'000'000'000, // timeout in nanoseconds
        vk.semaphore_swapchainImgAvailable[vk.frameInd],
        VK_NULL_HANDLE, // fence
        &imgIndex); // we will use this index for addressing the corresponding cmd buffer

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &vk.semaphore_swapchainImgAvailable[vk.frameInd];
    const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vk.cmdBuffers[vk.frameInd];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &vk.semaphore_drawFinished[vk.frameInd];

    vkWaitForFences(vk.device, 1, &vk.fence_queueWorkFinished[vk.frameInd], VK_TRUE, -1);
    vkResetFences(vk.device, 1, &vk.fence_queueWorkFinished[vk.frameInd]);
    if(vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, vk.fence_queueWorkFinished[vk.frameInd]) != VK_SUCCESS) {
        printf("Error submitting cmd buffers");
        exit(-1);
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &vk.semaphore_drawFinished[vk.frameInd];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &vk.swapchain;
    presentInfo.pImageIndices = &imgIndex;

    vkQueuePresentKHR(vk.presentationQueue, &presentInfo);

    vk.frameInd = (vk.frameInd + 1) % 2;
}

int main()
{
    printf("hello vulkan\n");

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // explicitly tell GLFW not to create an OpenGL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "hello vulkan", nullptr, nullptr);

    initVulkan();

    const clock_t startTime = clock();
    int frameCounter = 0;
    const int FRAME_COUNTER = 100000;

    while(!glfwWindowShouldClose(window))
    {
        //printf("frameCounter: %d\n", frameCounter);
        glfwPollEvents();
        draw();
        frameCounter++;
        if(frameCounter == FRAME_COUNTER)
            glfwSetWindowShouldClose(window, true);
    }

    const clock_t endTime = clock();
    const double elapsedSecs = double(endTime - startTime) / CLOCKS_PER_SEC;
    const double fps = FRAME_COUNTER / elapsedSecs;

    printf("elapsedSecs: %g\nfps: %g\n", elapsedSecs, fps);

    glfwDestroyWindow(window);
    glfwTerminate();
}