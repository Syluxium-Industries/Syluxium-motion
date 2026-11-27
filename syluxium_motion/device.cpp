// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

#include "syluxium_motion/device.hpp"

#include "flow_down_spv.hpp"
#include "flow_interp_spv.hpp"
#include "flow_luma_spv.hpp"
#include "flow_prop_spv.hpp"
#include "flow_search_spv.hpp"
#include "flow_search_wide_spv.hpp"
#include "flow_smooth_spv.hpp"
#include "flow_refine_spv.hpp"
#include "flow_still_spv.hpp"


#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>

namespace syluxium::motion {

namespace {

std::string uuidHex(const uint8_t* u) {
    std::string s;
    char b[3];
    for (int i = 0; i < 16; ++i) {
        std::snprintf(b, sizeof b, "%02x", u[i]);
        s += b;
    }
    return s;
}

struct EnvGuard {
    explicit EnvGuard(const std::string& name) : name_(name) {
        if (name_.empty()) return;
        const char* old = std::getenv(name_.c_str());
        had_ = old != nullptr;
        if (had_) prev_ = old;
        setenv(name_.c_str(), "1", 1);
    }
    ~EnvGuard() {
        if (name_.empty()) return;
        if (had_) setenv(name_.c_str(), prev_.c_str(), 1);
        else unsetenv(name_.c_str());
    }
    std::string name_, prev_;
    bool had_ = false;
};

VkDescriptorType vkType(DescriptorKind k) {
    switch (k) {
        case DescriptorKind::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorKind::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        case DescriptorKind::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case DescriptorKind::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

}

Device::Device(const DeviceOptions& opts)
    : performance_(opts.performance), interop_(opts.interop), wantHostImport_(opts.hostImport) {
    wantPresent_ = opts.present;
    EnvGuard guard(opts.disableLayerEnv);
    const auto& l = loader();

    uint32_t apiVersion = VK_API_VERSION_1_0;
    if (l.vkEnumerateInstanceVersion) l.vkEnumerateInstanceVersion(&apiVersion);
    if (apiVersion < VK_API_VERSION_1_2)
        throw std::runtime_error("the Vulkan loader is older than 1.2");
    compat_ = std::getenv("SYLUXIUM_MOTION_COMPAT") != nullptr;
    instanceApi_ = compat_ ? VK_API_VERSION_1_2 : std::min<uint32_t>(apiVersion, VK_API_VERSION_1_3);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Syluxium Motion";
    app.pEngineName = "Syluxium Motion";
    app.apiVersion = instanceApi_;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    std::vector<const char*> instExts;
    if (wantPresent_ && l.vkEnumerateInstanceExtensionProperties) {
        uint32_t n = 0;
        l.vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> props(n);
        l.vkEnumerateInstanceExtensionProperties(nullptr, &n, props.data());
        for (const char* want : {"VK_KHR_surface", "VK_KHR_xcb_surface", "VK_KHR_xlib_surface", "VK_KHR_wayland_surface",
                                 "VK_KHR_get_surface_capabilities2"})
            for (const auto& p : props)
                if (std::strcmp(p.extensionName, want) == 0) instExts.push_back(want);
    }
    ici.enabledExtensionCount = uint32_t(instExts.size());
    ici.ppEnabledExtensionNames = instExts.data();
    check(l.vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
    try {
        loadInstanceFns(instance_, vki_);
        pickPhysicalDevice(opts.gpuUuid);
        createDevice();
        buildPipelines();
    } catch (...) {
        cleanup();
        throw;
    }
}

Device::~Device() { cleanup(); }

void Device::cleanup() {
    if (device_) {
        vk_.vkDeviceWaitIdle(device_);
        for (auto s : samplers_)
            if (s) vk_.vkDestroySampler(device_, s, nullptr);
        for (auto& [n, p] : pipelines_) {
            if (p.pipeline) vk_.vkDestroyPipeline(device_, p.pipeline, nullptr);
            if (p.layout) vk_.vkDestroyPipelineLayout(device_, p.layout, nullptr);
            if (p.setLayout) vk_.vkDestroyDescriptorSetLayout(device_, p.setLayout, nullptr);
        }
        pipelines_.clear();
        vk_.vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vki_.vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void Device::pickPhysicalDevice(const std::string& uuid) {
    uint32_t n = 0;
    vki_.vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vki_.vkEnumeratePhysicalDevices(instance_, &n, pds.data());
    if (pds.empty()) throw std::runtime_error("no Vulkan GPUs");

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (auto pd : pds) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        vki_.vkGetPhysicalDeviceProperties2(pd, &p2);
        if (p2.properties.apiVersion < VK_API_VERSION_1_2) continue;
        if (!uuid.empty() && uuidHex(id.deviceUUID) == uuid) {
            physical_ = pd;
            break;
        }
        if (!fallback && p2.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) fallback = pd;
    }
    if (!physical_) {
        if (!uuid.empty()) std::fprintf(stderr, "syluxium-motion: GPU %s not found, using the first suitable GPU\n", uuid.c_str());
        physical_ = fallback;
    }
    if (!physical_) throw std::runtime_error("no GPU with Vulkan 1.2 or newer found");

    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vki_.vkGetPhysicalDeviceProperties2(physical_, &p2);
    name_ = p2.properties.deviceName;
    deviceApi_ = std::min(p2.properties.apiVersion, instanceApi_);
    std::memcpy(uuid_.data(), id.deviceUUID, 16);
    timestampPeriod_ = p2.properties.limits.timestampPeriod;
    uboAlign_ = std::max<VkDeviceSize>(p2.properties.limits.minUniformBufferOffsetAlignment, 16);
    vki_.vkGetPhysicalDeviceMemoryProperties(physical_, &memProps_);
}

void Device::createDevice() {
    uint32_t n = 0;
    vki_.vkEnumerateDeviceExtensionProperties(physical_, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    vki_.vkEnumerateDeviceExtensionProperties(physical_, nullptr, &n, props.data());
    std::set<std::string> haveExt;
    for (const auto& p : props) haveExt.insert(p.extensionName);

    std::vector<const char*> exts;
    for (const char* e : {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME}) {
        if (haveExt.count(e)) exts.push_back(e);
        else if (interop_) throw std::runtime_error(std::string(name_) + " lacks " + e);
    }
    if (wantHostImport_) {
        if (!haveExt.count(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
            throw std::runtime_error(name_ + " can't import host memory (VK_EXT_external_memory_host)");
        exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        hostImport_ = true;
    }
    const char* calibrated = haveExt.count("VK_KHR_calibrated_timestamps") ? "VK_KHR_calibrated_timestamps"
                             : haveExt.count("VK_EXT_calibrated_timestamps") ? "VK_EXT_calibrated_timestamps"
                                                                              : nullptr;
    bool timingExts = false;
    if (wantPresent_) {
        if (!haveExt.count(VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            throw std::runtime_error(name_ + " can't present (no VK_KHR_swapchain)");
        exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        timingExts = calibrated && haveExt.count("VK_EXT_present_timing") && haveExt.count("VK_KHR_present_id2");
    }
    const char* robustness = compat_ ? nullptr
                             : haveExt.count("VK_KHR_robustness2") ? "VK_KHR_robustness2"
                             : haveExt.count(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) ? VK_EXT_ROBUSTNESS_2_EXTENSION_NAME
                                                                                : nullptr;
    const bool core13 = deviceApi_ >= VK_API_VERSION_1_3;
    const bool sync2Ext = !core13 && !compat_ && haveExt.count(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

    VkPhysicalDeviceRobustness2FeaturesEXT haveRob{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceSynchronization2Features haveSync{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                                                      robustness ? static_cast<void*>(&haveRob) : nullptr};
    VkPhysicalDeviceVulkan13Features have13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &haveSync};
    void* after12 = core13 ? static_cast<void*>(&have13) : static_cast<void*>(&haveSync);
    if (!core13 && !sync2Ext) after12 = robustness ? static_cast<void*>(&haveRob) : nullptr;
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, after12};
    VkPhysicalDevicePresentTimingFeaturesEXT havePt{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT, &have12};
    VkPhysicalDevicePresentId2FeaturesKHR haveId2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR, &havePt};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, timingExts ? static_cast<void*>(&haveId2)
                                                                                           : static_cast<void*>(&have12)};
    vki_.vkGetPhysicalDeviceFeatures2(physical_, &have);
    timingExts = timingExts && havePt.presentTiming && haveId2.presentId2;
    nullDescriptor_ = robustness && haveRob.nullDescriptor;
    if (!have12.timelineSemaphore) throw std::runtime_error(name_ + " lacks timeline semaphores");
    sync2_ = core13 && have13.synchronization2 && !compat_ ? 0 : sync2Ext && haveSync.synchronization2 ? 1 : 2;
    if (nullDescriptor_) exts.push_back(robustness);
    if (sync2_ == 1) exts.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    if (!have.features.shaderStorageImageWriteWithoutFormat)
        throw std::runtime_error(name_ + " lacks shaderStorageImageWriteWithoutFormat");

    VkPhysicalDeviceRobustness2FeaturesEXT rob{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    rob.nullDescriptor = VK_TRUE;
    VkPhysicalDeviceSynchronization2Features syncF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                                                   nullDescriptor_ ? static_cast<void*>(&rob) : nullptr};
    syncF.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                         nullDescriptor_ ? static_cast<void*>(&rob) : nullptr};
    f13.synchronization2 = VK_TRUE;
    void* next12 = sync2_ == 0 ? static_cast<void*>(&f13)
                   : sync2_ == 1 ? static_cast<void*>(&syncF)
                                 : (nullDescriptor_ ? static_cast<void*>(&rob) : nullptr);
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, next12};
    f12.timelineSemaphore = VK_TRUE;
    VkPhysicalDevicePresentTimingFeaturesEXT pt{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT, &f12};
    pt.presentTiming = VK_TRUE;
    pt.presentAtRelativeTime = havePt.presentAtRelativeTime;
    VkPhysicalDevicePresentId2FeaturesKHR id2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR, &pt};
    id2.presentId2 = VK_TRUE;
    if (timingExts) {
        exts.push_back("VK_EXT_present_timing");
        exts.push_back("VK_KHR_present_id2");
        exts.push_back(calibrated);
        presentTiming_ = true;
    }
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, timingExts ? static_cast<void*>(&id2)
                                                                                        : static_cast<void*>(&f12)};
    f.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    f.features.shaderStorageImageReadWithoutFormat = have.features.shaderStorageImageReadWithoutFormat;
    f.features.shaderStorageImageExtendedFormats = have.features.shaderStorageImageExtendedFormats;

    vki_.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vki_.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &n, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < n && !found; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            family_ = i;
            found = true;
        }
    for (uint32_t i = 0; i < n && !found; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family_ = i;
            found = true;
        }
    if (!found) throw std::runtime_error(name_ + " has no compute queue");
    if (qf[family_].timestampValidBits == 0) timestampPeriod_ = 0;

    presentFamily_ = family_;
    if (wantPresent_ && !(qf[family_].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        bool gfx = false;
        for (uint32_t i = 0; i < n && !gfx; ++i)
            if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                presentFamily_ = i;
                gfx = true;
            }
        if (!gfx) throw std::runtime_error(name_ + " has no graphics queue to present from");
    }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qcis[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    qcis[0].queueFamilyIndex = family_;
    qcis[0].queueCount = 1;
    qcis[0].pQueuePriorities = &prio;
    qcis[1].queueFamilyIndex = presentFamily_;
    qcis[1].queueCount = 1;
    qcis[1].pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f};
    dci.queueCreateInfoCount = presentFamily_ != family_ ? 2 : 1;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = uint32_t(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    check(vki_.vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice (frame generation)");
    loadDeviceFns(vki_, device_, vk_, sync2_);
    vk_.vkGetDeviceQueue(device_, family_, 0, &queue_);
    if (wantPresent_) {
        vk_.vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
        present_ = vk_.vkCreateSwapchainKHR && vk_.vkQueuePresentKHR;
        if (!present_) throw std::runtime_error(name_ + ": swapchain functions missing");
    }

    auto makeSampler = [&](VkSamplerAddressMode mode, VkBorderColor border) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = mode;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        sci.borderColor = border;
        VkSampler s;
        check(vk_.vkCreateSampler(device_, &sci, nullptr, &s), "vkCreateSampler");
        return s;
    };
    samplers_[size_t(SamplerKind::BorderBlack)] =
        makeSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
    samplers_[size_t(SamplerKind::BorderWhite)] =
        makeSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE);
    samplers_[size_t(SamplerKind::Edge)] =
        makeSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
}

void Device::addPipeline(const std::string& name, const uint32_t* code, size_t words,
                         std::vector<ShaderBinding> bindings, std::array<uint32_t, 3> workgroup) {
    Pipeline p;
    p.bindings = std::move(bindings);
    p.workgroup = workgroup;
    std::vector<VkDescriptorSetLayoutBinding> lb;
    for (const auto& b : p.bindings)
        lb.push_back({b.binding, vkType(b.kind), 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = uint32_t(lb.size());
    dsl.pBindings = lb.data();
    check(vk_.vkCreateDescriptorSetLayout(device_, &dsl, nullptr, &p.setLayout), "descriptor set layout");
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.setLayout;
    check(vk_.vkCreatePipelineLayout(device_, &pl, nullptr, &p.layout), "pipeline layout");

    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = words * 4;
    sm.pCode = code;
    VkShaderModule mod;
    check(vk_.vkCreateShaderModule(device_, &sm, nullptr, &mod), "shader module");
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName = "main";
    cp.layout = p.layout;
    const VkResult r = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipeline);
    vk_.vkDestroyShaderModule(device_, mod, nullptr);
    if (r != VK_SUCCESS) {
        vk_.vkDestroyPipelineLayout(device_, p.layout, nullptr);
        vk_.vkDestroyDescriptorSetLayout(device_, p.setLayout, nullptr);
    }
    check(r, ("compute pipeline " + name).c_str());
    pipelines_.emplace(name, std::move(p));
}

void Device::buildPipelines() {
    using K = DescriptorKind;
    auto spec = [](std::initializer_list<K> kinds) {
        std::vector<ShaderBinding> b;
        uint32_t n = 0;
        for (K k : kinds) b.push_back({k, n++});
        return b;
    };
    const std::array<uint32_t, 3> wg{8, 8, 1};
    addPipeline("flow.luma", kFlow_luma, std::size(kFlow_luma), spec({K::UniformBuffer, K::Sampler, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.down", kFlow_down, std::size(kFlow_down), spec({K::Sampler, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.search", kFlow_search, std::size(kFlow_search),
                spec({K::Sampler, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.search.wide", kFlow_search_wide, std::size(kFlow_search_wide),
                spec({K::Sampler, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.prop", kFlow_prop, std::size(kFlow_prop),
                spec({K::Sampler, K::SampledImage, K::SampledImage, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.smooth", kFlow_smooth, std::size(kFlow_smooth), spec({K::Sampler, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.refine", kFlow_refine, std::size(kFlow_refine),
                spec({K::Sampler, K::SampledImage, K::SampledImage, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.interp", kFlow_interp, std::size(kFlow_interp),
                spec({K::UniformBuffer, K::Sampler, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::SampledImage, K::StorageImage}), wg);
    addPipeline("flow.still", kFlow_still, std::size(kFlow_still),
                spec({K::UniformBuffer, K::Sampler, K::SampledImage, K::SampledImage, K::SampledImage, K::StorageImage}), wg);
}

std::string_view descriptorKindName(DescriptorKind k) {
    switch (k) {
        case DescriptorKind::UniformBuffer: return "uniform buffer";
        case DescriptorKind::Sampler: return "sampler";
        case DescriptorKind::SampledImage: return "sampled image";
        case DescriptorKind::StorageImage: return "storage image";
    }
    return "?";
}

const Pipeline& Device::pipeline(const std::string& name) const {
    auto it = pipelines_.find(name);
    if (it == pipelines_.end()) throw std::runtime_error("no pipeline " + name);
    return it->second;
}

uint32_t Device::memoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const {
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (memProps_.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("no suitable memory type");
}

}
