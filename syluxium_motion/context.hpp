// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

#pragma once

#include "syluxium_motion/device.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <string>
#include <vector>

namespace syluxium::motion {

struct ContextOptions {
    VkExtent2D extent{};
    bool hdr = false;
    float flowScale = 1.0f;
    uint32_t generated = 1;
    bool exportable = false;
    VkFormat frameFormat = VK_FORMAT_UNDEFINED;
    bool depth = false;
    bool depthReversed = false;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkDeviceSize size = 0;
    uint32_t memoryType = 0;
    VkImageUsageFlags usage = 0;
};

class Context {
public:
    Context(Device& device, const ContextOptions& opts);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    const ContextOptions& options() const { return opts_; }
    VkFormat frameFormat() const { return frameFormat_; }

    uint64_t stride() const { return opts_.generated + 1; }
    uint64_t inputReady(uint64_t k) const { return k * stride() + 1; }
    uint64_t generated(uint64_t k, uint32_t i) const { return k * stride() + 2 + i; }

    void submit(uint64_t k);
    void submit(uint64_t k, uint32_t count, const float* times);

    double lastGpuMs() const { return lastGpuMs_; }

    int exportInput(int i) const;
    int exportDepthInput(int i) const;
    void setDepthMode(uint32_t mode);
    int exportOutput(uint32_t i) const;
    int exportTimeline() const;
    int exportConsumed() const;
    const Image& input(int i) const { return inputs_[i]; }
    const Image& depthInput(int i) const { return depthInputs_[i]; }
    VkSemaphore timelineSemaphore() const { return timeline_; }
    VkSemaphore consumedSemaphore() const { return consumed_; }
    const Image& output(uint32_t i) const { return outputs_[i]; }

    void testUpload(uint64_t k, const std::vector<uint8_t>& rgba, const std::vector<float>* depth = nullptr);
    std::vector<uint8_t> testDownload(uint64_t k, uint32_t i);
    void testConsumed(uint64_t k);

    size_t dispatchCount() const { return dispatchesPerFrame_; }

    std::vector<uint8_t> debugDownload(const Image& img);

    std::vector<std::string> debugStats();

private:
    struct Bind {
        DescriptorKind kind;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDeviceSize uboOffset = 0;
    };
    struct Dispatch {
        const Pipeline* pipeline;
        std::string label;
        std::vector<Bind> binds;
        uint32_t groupsX, groupsY;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };
    class Binds;

    Image* makeImage(VkExtent2D e, VkFormat f, bool exportable = false);
    VkDeviceSize uboSlot(float timestamp, bool unique = false);
    void add(std::vector<Dispatch>& list, const std::string& shader, const Binds& b, VkExtent2D e, uint32_t div);
    void buildFlowGraph();
    void allocateSets();
    void record();
    void initLayouts();
    VkCommandBuffer oneShotBegin();
    void oneShotSubmit(VkCommandBuffer cb, const VkSemaphoreSubmitInfo* wait, const VkSemaphoreSubmitInfo* signal);
    void readTiming(uint64_t k);

    Device& dev_;
    ContextOptions opts_;
    VkFormat frameFormat_;

    std::vector<std::unique_ptr<Image>> images_;
    std::array<Image, 2> inputs_{};
    std::array<Image, 2> depthInputs_{};
    Image* zero_ = nullptr;
    std::vector<Image> outputs_;

    VkBuffer ubo_ = VK_NULL_HANDLE;
    VkDeviceMemory uboMem_ = VK_NULL_HANDLE;
    uint8_t* uboMap_ = nullptr;
    std::vector<std::pair<float, VkDeviceSize>> uboSlots_;

    VkSemaphore timeline_ = VK_NULL_HANDLE;
    VkSemaphore consumed_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    VkQueryPool prof_ = VK_NULL_HANDLE;
    std::vector<std::string> profLabels_;
    VkFence oneShotFence_ = VK_NULL_HANDLE;

    static constexpr uint32_t kPhases = 6;
    std::array<std::vector<Dispatch>, kPhases> pre_;
    std::array<std::vector<std::vector<Dispatch>>, kPhases> main_;
    std::array<VkCommandBuffer, kPhases> preCmd_{};
    std::array<std::vector<VkCommandBuffer>, kPhases> mainCmd_;
    std::array<std::vector<VkDeviceSize>, kPhases> timeSlots_;
    size_t dispatchesPerFrame_ = 0;
    std::vector<std::pair<std::string, Image*>> debugImages_;
    double lastGpuMs_ = 0;
    uint64_t submitted_ = 0;
};

}
