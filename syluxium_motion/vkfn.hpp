// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <stdexcept>
#include <string>

namespace syluxium::motion {

#define SYLUXIUM_MOTION_VK_INSTANCE_FNS(X)              \
    X(vkDestroyInstance)                       \
    X(vkEnumeratePhysicalDevices)              \
    X(vkGetPhysicalDeviceProperties)           \
    X(vkGetPhysicalDeviceProperties2)          \
    X(vkGetPhysicalDeviceFeatures2)            \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties)     \
    X(vkGetPhysicalDeviceFormatProperties)     \
    X(vkEnumerateDeviceExtensionProperties)    \
    X(vkCreateDevice)                          \
    X(vkGetDeviceProcAddr)

#define SYLUXIUM_MOTION_VK_DEVICE_FNS(X)        \
    X(vkDestroyDevice)                 \
    X(vkGetDeviceQueue)                \
    X(vkDeviceWaitIdle)                \
    X(vkQueueWaitIdle)                 \
    X(vkCreateImage)                   \
    X(vkDestroyImage)                  \
    X(vkGetImageMemoryRequirements)    \
    X(vkCreateImageView)               \
    X(vkDestroyImageView)              \
    X(vkCreateBuffer)                  \
    X(vkDestroyBuffer)                 \
    X(vkGetBufferMemoryRequirements)   \
    X(vkAllocateMemory)                \
    X(vkFreeMemory)                    \
    X(vkBindImageMemory)               \
    X(vkBindBufferMemory)              \
    X(vkMapMemory)                     \
    X(vkUnmapMemory)                   \
    X(vkCreateSampler)                 \
    X(vkDestroySampler)                \
    X(vkCreateShaderModule)            \
    X(vkDestroyShaderModule)           \
    X(vkCreateDescriptorSetLayout)     \
    X(vkDestroyDescriptorSetLayout)    \
    X(vkCreatePipelineLayout)          \
    X(vkDestroyPipelineLayout)         \
    X(vkCreateComputePipelines)        \
    X(vkDestroyPipeline)               \
    X(vkCreateDescriptorPool)          \
    X(vkDestroyDescriptorPool)         \
    X(vkAllocateDescriptorSets)        \
    X(vkUpdateDescriptorSets)          \
    X(vkCreateCommandPool)             \
    X(vkDestroyCommandPool)            \
    X(vkAllocateCommandBuffers)        \
    X(vkBeginCommandBuffer)            \
    X(vkResetCommandBuffer)            \
    X(vkEndCommandBuffer)              \
    X(vkCmdBindPipeline)               \
    X(vkCmdBindDescriptorSets)         \
    X(vkCmdDispatch)                   \
    X(vkCmdCopyBufferToImage)          \
    X(vkCmdCopyImageToBuffer)          \
    X(vkCmdClearColorImage)            \
    X(vkCmdResetQueryPool)             \
    X(vkCreateQueryPool)               \
    X(vkDestroyQueryPool)              \
    X(vkGetQueryPoolResults)           \
    X(vkCreateSemaphore)               \
    X(vkDestroySemaphore)              \
    X(vkWaitSemaphores)                \
    X(vkSignalSemaphore)               \
    X(vkGetSemaphoreCounterValue)      \
    X(vkCreateFence)                   \
    X(vkDestroyFence)                  \
    X(vkWaitForFences)                 \
    X(vkResetFences)
#define SYLUXIUM_MOTION_VK_DEVICE_OPTIONAL_FNS(X) \
    X(vkGetMemoryFdKHR)                \
    X(vkGetSemaphoreFdKHR)             \
    X(vkGetMemoryHostPointerPropertiesEXT)

#define SYLUXIUM_MOTION_VK_INSTANCE_OPTIONAL_FNS(X)          \
    X(vkDestroySurfaceKHR)                          \
    X(vkGetPhysicalDeviceSurfaceSupportKHR)         \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)    \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR)         \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR)    \
    X(vkGetPhysicalDeviceSurfaceCapabilities2KHR)
#define SYLUXIUM_MOTION_VK_DEVICE_PRESENT_FNS(X) \
    X(vkSetSwapchainPresentTimingQueueSizeEXT) \
    X(vkGetSwapchainTimingPropertiesEXT) \
    X(vkGetPastPresentationTimingEXT) \
    X(vkCreateSwapchainKHR)             \
    X(vkDestroySwapchainKHR)            \
    X(vkGetSwapchainImagesKHR)          \
    X(vkAcquireNextImageKHR)            \
    X(vkQueuePresentKHR)                \
    X(vkCmdCopyImage)                   \
    X(vkCmdBlitImage)

#define SYLUXIUM_MOTION_VK_DECLARE(name) PFN_##name name = nullptr;

struct InstanceFns {
    VkInstance instance = VK_NULL_HANDLE;
    SYLUXIUM_MOTION_VK_INSTANCE_FNS(SYLUXIUM_MOTION_VK_DECLARE)
    SYLUXIUM_MOTION_VK_INSTANCE_OPTIONAL_FNS(SYLUXIUM_MOTION_VK_DECLARE)
    PFN_vkVoidFunction vkCreateXcbSurfaceKHR = nullptr;
    PFN_vkVoidFunction vkCreateXlibSurfaceKHR = nullptr;
    PFN_vkVoidFunction vkCreateWaylandSurfaceKHR = nullptr;
};
struct DeviceFns {
    SYLUXIUM_MOTION_VK_DEVICE_FNS(SYLUXIUM_MOTION_VK_DECLARE)
    PFN_vkQueueSubmit2 vkQueueSubmit2 = nullptr;
    PFN_vkCmdPipelineBarrier2 vkCmdPipelineBarrier2 = nullptr;
    PFN_vkCmdWriteTimestamp2 vkCmdWriteTimestamp2 = nullptr;
    SYLUXIUM_MOTION_VK_DEVICE_OPTIONAL_FNS(SYLUXIUM_MOTION_VK_DECLARE)
    SYLUXIUM_MOTION_VK_DEVICE_PRESENT_FNS(SYLUXIUM_MOTION_VK_DECLARE)
};

struct Loader {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = nullptr;
};

const Loader& loader();

void loadInstanceFns(VkInstance instance, InstanceFns& out);
void loadDeviceFns(const InstanceFns& inst, VkDevice device, DeviceFns& out, int sync2 = 0);

struct VulkanError : std::runtime_error {
    VulkanError(VkResult r, const std::string& what)
        : std::runtime_error(what + " (VkResult " + std::to_string(int(r)) + ")"), result(r) {}
    VkResult result;
};

inline void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw VulkanError(r, what);
}

}
