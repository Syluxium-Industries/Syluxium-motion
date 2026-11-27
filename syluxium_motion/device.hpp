// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

#pragma once

#include "syluxium_motion/vkfn.hpp"

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace syluxium::motion {

enum class DescriptorKind : uint8_t { UniformBuffer = 0, Sampler = 1, SampledImage = 2, StorageImage = 3 };
std::string_view descriptorKindName(DescriptorKind k);

struct ShaderBinding {
    DescriptorKind kind;
    uint32_t binding = 0;
};

struct Pipeline {
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<ShaderBinding> bindings;
    std::array<uint32_t, 3> workgroup{};
};

enum class SamplerKind { BorderBlack, BorderWhite, Edge };

struct DeviceOptions {
    std::string gpuUuid;
    bool performance = false;
    bool interop = true;
    bool hostImport = false;
    bool present = false;
    std::string disableLayerEnv;
};

class Device {
public:
    explicit Device(const DeviceOptions& opts);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const InstanceFns& vki() const { return vki_; }
    const DeviceFns& vk() const { return vk_; }
    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical() const { return physical_; }
    VkDevice handle() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamily() const { return family_; }
    const std::string& name() const { return name_; }
    const std::array<uint8_t, 16>& deviceUuid() const { return uuid_; }
    bool performance() const { return performance_; }
    float timestampPeriodNs() const { return timestampPeriod_; }
    VkDeviceSize uniformAlignment() const { return uboAlign_; }
    bool hostImport() const { return hostImport_; }
    bool nullDescriptor() const { return nullDescriptor_; }
    uint32_t apiVersion() const { return deviceApi_; }
    int sync2Mode() const { return sync2_; }
    bool canPresent() const { return present_; }
    bool presentTiming() const { return presentTiming_ && vk_.vkGetPastPresentationTimingEXT; }
    VkQueue presentQueue() const { return presentQueue_; }
    uint32_t presentFamily() const { return presentFamily_; }
    std::vector<uint32_t> frameFamilies() const {
        return present_ && presentFamily_ != family_ ? std::vector<uint32_t>{family_, presentFamily_}
                                                     : std::vector<uint32_t>{family_};
    }
    std::mutex& submitLock() const { return submitLock_; }

    const Pipeline& pipeline(const std::string& name) const;
    VkSampler sampler(SamplerKind k) const { return samplers_[size_t(k)]; }

    uint32_t memoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const;

private:
    void cleanup();
    void pickPhysicalDevice(const std::string& uuid);
    void createDevice();
    void buildPipelines();
    void addPipeline(const std::string& name, const uint32_t* code, size_t words, std::vector<ShaderBinding> bindings,
                     std::array<uint32_t, 3> workgroup);

    InstanceFns vki_;
    DeviceFns vk_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t family_ = 0;
    std::string name_;
    std::array<uint8_t, 16> uuid_{};
    bool performance_ = false;
    bool interop_ = true;
    bool compat_ = false, nullDescriptor_ = false;
    int sync2_ = 0;
    uint32_t instanceApi_ = 0, deviceApi_ = 0;
    bool hostImport_ = false;
    bool wantPresent_ = false;
    bool present_ = false;
    bool presentTiming_ = false;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    uint32_t presentFamily_ = 0;
    mutable std::mutex submitLock_;
    bool wantHostImport_ = false;
    float timestampPeriod_ = 1.0f;
    VkDeviceSize uboAlign_ = 256;
    VkPhysicalDeviceMemoryProperties memProps_{};
    std::map<std::string, Pipeline> pipelines_;
    std::array<VkSampler, 3> samplers_{};
};

}
