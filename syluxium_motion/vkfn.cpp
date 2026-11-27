// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

#include "syluxium_motion/vkfn.hpp"
#include <vector>
#include <stdexcept>

#include <dlfcn.h>

#include <mutex>

namespace syluxium::motion {

const Loader& loader() {
    static Loader l;
    static std::once_flag once;
    static std::string error;
    std::call_once(once, [] {
        void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            error = std::string("cannot load libvulkan.so.1: ") + dlerror();
            return;
        }
        l.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
        if (!l.vkGetInstanceProcAddr) {
            error = "libvulkan has no vkGetInstanceProcAddr";
            return;
        }
        l.vkCreateInstance =
            reinterpret_cast<PFN_vkCreateInstance>(l.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
        l.vkEnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            l.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        l.vkEnumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            l.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    });
    if (!l.vkCreateInstance)
        throw std::runtime_error(error.empty() ? "Vulkan loader unavailable" : error);
    return l;
}

void loadInstanceFns(VkInstance instance, InstanceFns& out) {
    const auto gipa = loader().vkGetInstanceProcAddr;
    out.instance = instance;
#define LOAD(name)                                                                  \
    out.name = reinterpret_cast<PFN_##name>(gipa(instance, #name));                 \
    if (!out.name) throw std::runtime_error("missing Vulkan instance function " #name);
    SYLUXIUM_MOTION_VK_INSTANCE_FNS(LOAD)
#undef LOAD
#define LOAD_OPTIONAL(name) out.name = reinterpret_cast<PFN_##name>(gipa(instance, #name));
    SYLUXIUM_MOTION_VK_INSTANCE_OPTIONAL_FNS(LOAD_OPTIONAL)
#undef LOAD_OPTIONAL
    out.vkCreateXcbSurfaceKHR = gipa(instance, "vkCreateXcbSurfaceKHR");
    out.vkCreateXlibSurfaceKHR = gipa(instance, "vkCreateXlibSurfaceKHR");
    out.vkCreateWaylandSurfaceKHR = gipa(instance, "vkCreateWaylandSurfaceKHR");
}

namespace {

PFN_vkQueueSubmit g_submit = nullptr;
PFN_vkCmdPipelineBarrier g_barrier = nullptr;
PFN_vkCmdWriteTimestamp g_timestamp = nullptr;

VkPipelineStageFlags stage1(VkPipelineStageFlags2 s, VkPipelineStageFlags none) {
    if (s == 0) return none;
    VkPipelineStageFlags r = VkPipelineStageFlags(s & 0xffffffffull);
    if (s >> 32) {
        constexpr VkPipelineStageFlags2 transfer = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                                                   VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
        r |= (s & transfer) ? VK_PIPELINE_STAGE_TRANSFER_BIT : 0;
        if (s & ~(transfer | 0xffffffffull)) r |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    return r ? r : none;
}
VkAccessFlags access1(VkAccessFlags2 a) {
    VkAccessFlags r = VkAccessFlags(a & 0xffffffffull);
    if (a & (VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT)) r |= VK_ACCESS_SHADER_READ_BIT;
    if (a & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) r |= VK_ACCESS_SHADER_WRITE_BIT;
    return r;
}

VKAPI_ATTR void VKAPI_CALL barrier2Shim(VkCommandBuffer cb, const VkDependencyInfo* d) {
    for (uint32_t i = 0; i < d->memoryBarrierCount; ++i) {
        const auto& m = d->pMemoryBarriers[i];
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, access1(m.srcAccessMask), access1(m.dstAccessMask)};
        g_barrier(cb, stage1(m.srcStageMask, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT),
                  stage1(m.dstStageMask, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT), d->dependencyFlags, 1, &b, 0, nullptr, 0,
                  nullptr);
    }
    for (uint32_t i = 0; i < d->bufferMemoryBarrierCount; ++i) {
        const auto& m = d->pBufferMemoryBarriers[i];
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, access1(m.srcAccessMask),
                                access1(m.dstAccessMask), m.srcQueueFamilyIndex, m.dstQueueFamilyIndex, m.buffer, m.offset,
                                m.size};
        g_barrier(cb, stage1(m.srcStageMask, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT),
                  stage1(m.dstStageMask, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT), d->dependencyFlags, 0, nullptr, 1, &b, 0,
                  nullptr);
    }
    for (uint32_t i = 0; i < d->imageMemoryBarrierCount; ++i) {
        const auto& m = d->pImageMemoryBarriers[i];
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, access1(m.srcAccessMask),
                               access1(m.dstAccessMask), m.oldLayout, m.newLayout, m.srcQueueFamilyIndex,
                               m.dstQueueFamilyIndex, m.image, m.subresourceRange};
        g_barrier(cb, stage1(m.srcStageMask, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT),
                  stage1(m.dstStageMask, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT), d->dependencyFlags, 0, nullptr, 0, nullptr,
                  1, &b);
    }
}

VKAPI_ATTR void VKAPI_CALL timestamp2Shim(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool, uint32_t q) {
    const VkPipelineStageFlags s = stage1(stage, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    g_timestamp(cb, (s & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                pool, q);
}

VKAPI_ATTR VkResult VKAPI_CALL submit2Shim(VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence) {
    struct Batch {
        std::vector<VkSemaphore> waits, signals;
        std::vector<uint64_t> waitValues, signalValues;
        std::vector<VkPipelineStageFlags> stages;
        std::vector<VkCommandBuffer> cbs;
        VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    };
    std::vector<Batch> batches(count);
    std::vector<VkSubmitInfo> infos(count, {VK_STRUCTURE_TYPE_SUBMIT_INFO});
    for (uint32_t i = 0; i < count; ++i) {
        const VkSubmitInfo2& s = submits[i];
        Batch& b = batches[i];
        for (uint32_t j = 0; j < s.waitSemaphoreInfoCount; ++j) {
            b.waits.push_back(s.pWaitSemaphoreInfos[j].semaphore);
            b.waitValues.push_back(s.pWaitSemaphoreInfos[j].value);
            b.stages.push_back(stage1(s.pWaitSemaphoreInfos[j].stageMask, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT));
        }
        for (uint32_t j = 0; j < s.commandBufferInfoCount; ++j) b.cbs.push_back(s.pCommandBufferInfos[j].commandBuffer);
        for (uint32_t j = 0; j < s.signalSemaphoreInfoCount; ++j) {
            b.signals.push_back(s.pSignalSemaphoreInfos[j].semaphore);
            b.signalValues.push_back(s.pSignalSemaphoreInfos[j].value);
        }
        b.tl.waitSemaphoreValueCount = uint32_t(b.waitValues.size());
        b.tl.pWaitSemaphoreValues = b.waitValues.data();
        b.tl.signalSemaphoreValueCount = uint32_t(b.signalValues.size());
        b.tl.pSignalSemaphoreValues = b.signalValues.data();
        VkSubmitInfo& v = infos[i];
        v.pNext = &b.tl;
        v.waitSemaphoreCount = uint32_t(b.waits.size());
        v.pWaitSemaphores = b.waits.data();
        v.pWaitDstStageMask = b.stages.data();
        v.commandBufferCount = uint32_t(b.cbs.size());
        v.pCommandBuffers = b.cbs.data();
        v.signalSemaphoreCount = uint32_t(b.signals.size());
        v.pSignalSemaphores = b.signals.data();
    }
    return g_submit(queue, count, infos.data(), fence);
}

}

void loadDeviceFns(const InstanceFns& inst, VkDevice device, DeviceFns& out, int sync2) {
#define LOAD(name)                                                                  \
    out.name = reinterpret_cast<PFN_##name>(inst.vkGetDeviceProcAddr(device, #name)); \
    if (!out.name) throw std::runtime_error("missing Vulkan device function " #name);
    SYLUXIUM_MOTION_VK_DEVICE_FNS(LOAD)
#undef LOAD
#define LOAD_OPTIONAL(name) out.name = reinterpret_cast<PFN_##name>(inst.vkGetDeviceProcAddr(device, #name));
    SYLUXIUM_MOTION_VK_DEVICE_OPTIONAL_FNS(LOAD_OPTIONAL)
    SYLUXIUM_MOTION_VK_DEVICE_PRESENT_FNS(LOAD_OPTIONAL)
#undef LOAD_OPTIONAL
    if (sync2 == 0) {
        out.vkQueueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2>(inst.vkGetDeviceProcAddr(device, "vkQueueSubmit2"));
        out.vkCmdPipelineBarrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(inst.vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2"));
        out.vkCmdWriteTimestamp2 = reinterpret_cast<PFN_vkCmdWriteTimestamp2>(inst.vkGetDeviceProcAddr(device, "vkCmdWriteTimestamp2"));
    } else if (sync2 == 1) {
        out.vkQueueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2>(inst.vkGetDeviceProcAddr(device, "vkQueueSubmit2KHR"));
        out.vkCmdPipelineBarrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(inst.vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2KHR"));
        out.vkCmdWriteTimestamp2 = reinterpret_cast<PFN_vkCmdWriteTimestamp2>(inst.vkGetDeviceProcAddr(device, "vkCmdWriteTimestamp2KHR"));
    } else {
        const auto gipa = loader().vkGetInstanceProcAddr;
        g_submit = reinterpret_cast<PFN_vkQueueSubmit>(gipa(inst.instance, "vkQueueSubmit"));
        g_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(gipa(inst.instance, "vkCmdPipelineBarrier"));
        g_timestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(gipa(inst.instance, "vkCmdWriteTimestamp"));
        if (!g_submit || !g_barrier || !g_timestamp) throw std::runtime_error("missing classic Vulkan submit/barrier functions");
        out.vkQueueSubmit2 = submit2Shim;
        out.vkCmdPipelineBarrier2 = barrier2Shim;
        out.vkCmdWriteTimestamp2 = timestamp2Shim;
    }
    if (!out.vkQueueSubmit2 || !out.vkCmdPipelineBarrier2 || !out.vkCmdWriteTimestamp2) throw std::runtime_error("missing synchronization2 functions");
}

}
