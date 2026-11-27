// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

#include "syluxium_motion/context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <map>
#include <vector>

namespace syluxium::motion {

namespace {

constexpr VkFormat kRGBA8 = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kRGBA16F = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kR8 = VK_FORMAT_R8_UNORM;
constexpr VkFormat kR32F = VK_FORMAT_R32_SFLOAT;
constexpr VkFormat kRG32F = VK_FORMAT_R32G32_SFLOAT;
constexpr VkFormat kRG16F = VK_FORMAT_R16G16_SFLOAT;

struct Constants {
    uint32_t hdr;
    float timestamp;
    uint32_t depth;
    float pad;
};
static_assert(sizeof(Constants) == 16);
constexpr VkDeviceSize kUboSlots = 64;

VkExtent2D halve(VkExtent2D e) { return {(e.width + 1) >> 1, (e.height + 1) >> 1}; }

VkDescriptorType vkType(DescriptorKind k) {
    switch (k) {
        case DescriptorKind::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorKind::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        case DescriptorKind::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case DescriptorKind::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

using Group = std::vector<Image*>;

}

class Context::Binds {
public:
    explicit Binds(Context& c) : c_(c) {}
    Binds& ubo(VkDeviceSize offset) { b.push_back({DescriptorKind::UniformBuffer, {}, {}, offset}); return *this; }
    Binds& smp(SamplerKind k) { b.push_back({DescriptorKind::Sampler, {}, c_.dev_.sampler(k)}); return *this; }
    Binds& tex(const Image* i) {
        b.push_back({DescriptorKind::SampledImage, i ? i->view : c_.dev_.nullDescriptor() ? VK_NULL_HANDLE : c_.zero_->view});
        return *this;
    }
    Binds& tex(const Group& g) { for (auto* i : g) tex(i); return *this; }
    Binds& out(const Image* i) { b.push_back({DescriptorKind::StorageImage, i->view}); return *this; }
    Binds& out(const Group& g) { for (auto* i : g) out(i); return *this; }
    std::vector<Bind> b;

private:
    Context& c_;
};

Context::Context(Device& device, const ContextOptions& opts) : dev_(device), opts_(opts) {
    if (opts_.generated < 1 || opts_.generated > 7) throw std::runtime_error("generated frames must be 1..7");
    if (opts_.extent.width < 64 || opts_.extent.height < 64) throw std::runtime_error("frame too small");
    opts_.flowScale = std::clamp(opts_.flowScale, 0.25f, 1.0f);
    frameFormat_ = opts_.frameFormat != VK_FORMAT_UNDEFINED ? opts_.frameFormat : (opts_.hdr ? kRGBA16F : kRGBA8);
    const auto& vk = dev_.vk();
    const VkDevice d = dev_.handle();
    {
        VkFormatProperties fp{};
        dev_.vki().vkGetPhysicalDeviceFormatProperties(dev_.physical(), frameFormat_, &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                          VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((fp.optimalTilingFeatures & need) != need)
            throw std::runtime_error(dev_.name() + " can't use frame format " + std::to_string(int(frameFormat_)) +
                                     " for frame generation");
    }

    const VkDeviceSize align = dev_.uniformAlignment();
    const VkDeviceSize uboSize = kUboSlots * ((sizeof(Constants) + align - 1) / align * align);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = uboSize;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    check(vk.vkCreateBuffer(d, &bci, nullptr, &ubo_), "uniform buffer");
    VkMemoryRequirements mr;
    vk.vkGetBufferMemoryRequirements(d, ubo_, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex =
        dev_.memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vk.vkAllocateMemory(d, &mai, nullptr, &uboMem_), "uniform memory");
    check(vk.vkBindBufferMemory(d, ubo_, uboMem_, 0), "bind uniform memory");
    check(vk.vkMapMemory(d, uboMem_, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&uboMap_)), "map uniforms");

    auto makeTimeline = [&](VkSemaphore* out) {
        VkExportSemaphoreCreateInfo exp{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        exp.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, opts_.exportable ? &exp : nullptr};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        check(vk.vkCreateSemaphore(d, &sci, nullptr, out), "timeline semaphore");
    };
    makeTimeline(&timeline_);
    makeTimeline(&consumed_);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = dev_.queueFamily();
    check(vk.vkCreateCommandPool(d, &cpi, nullptr, &cmdPool_), "command pool");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vk.vkCreateFence(d, &fci, nullptr, &oneShotFence_), "fence");
    if (dev_.timestampPeriodNs() > 0) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2 * kPhases;
        check(vk.vkCreateQueryPool(d, &qpi, nullptr, &queries_), "query pool");
    }

    buildFlowGraph();
    allocateSets();
    initLayouts();
    record();
}

Context::~Context() {
    const auto& vk = dev_.vk();
    const VkDevice d = dev_.handle();
    vk.vkQueueWaitIdle(dev_.queue());
    if (queries_) vk.vkDestroyQueryPool(d, queries_, nullptr);
    if (prof_) vk.vkDestroyQueryPool(d, prof_, nullptr);
    if (oneShotFence_) vk.vkDestroyFence(d, oneShotFence_, nullptr);
    if (cmdPool_) vk.vkDestroyCommandPool(d, cmdPool_, nullptr);
    if (pool_) vk.vkDestroyDescriptorPool(d, pool_, nullptr);
    if (timeline_) vk.vkDestroySemaphore(d, timeline_, nullptr);
    if (consumed_) vk.vkDestroySemaphore(d, consumed_, nullptr);
    if (ubo_) vk.vkDestroyBuffer(d, ubo_, nullptr);
    if (uboMem_) vk.vkFreeMemory(d, uboMem_, nullptr);
    auto destroy = [&](Image& i) {
        if (i.view) vk.vkDestroyImageView(d, i.view, nullptr);
        if (i.image) vk.vkDestroyImage(d, i.image, nullptr);
        if (i.memory) vk.vkFreeMemory(d, i.memory, nullptr);
        i = {};
    };
    for (auto& i : images_) destroy(*i);
}

Image* Context::makeImage(VkExtent2D e, VkFormat f, bool exportable) {
    auto img = std::make_unique<Image>();
    const auto& vk = dev_.vk();
    const VkDevice d = dev_.handle();
    img->extent = e;
    img->format = f;

    VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, exportable ? &ext : nullptr};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = f;
    ici.extent = {e.width, e.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const auto families = dev_.frameFamilies();
    if (families.size() > 1 && (f == frameFormat_)) {
        ici.sharingMode = VK_SHARING_MODE_CONCURRENT;
        ici.queueFamilyIndexCount = uint32_t(families.size());
        ici.pQueueFamilyIndices = families.data();
    } else {
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    check(vk.vkCreateImage(d, &ici, nullptr, &img->image), "image");

    VkMemoryRequirements mr;
    vk.vkGetImageMemoryRequirements(d, img->image, &mr);
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.image = img->image;
    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &ded};
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, exportable ? static_cast<void*>(&exp) : nullptr};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = dev_.memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vk.vkAllocateMemory(d, &mai, nullptr, &img->memory), "image memory");
    check(vk.vkBindImageMemory(d, img->image, img->memory, 0), "bind image memory");
    img->size = mr.size;
    img->memoryType = mai.memoryTypeIndex;
    img->usage = ici.usage;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img->image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = f;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vk.vkCreateImageView(d, &vci, nullptr, &img->view), "image view");

    images_.push_back(std::move(img));
    return images_.back().get();
}

VkDeviceSize Context::uboSlot(float timestamp, bool unique) {
    if (!unique)
        for (const auto& [t, off] : uboSlots_)
            if (t == timestamp) return off;
    const VkDeviceSize align = dev_.uniformAlignment();
    const VkDeviceSize stride = (sizeof(Constants) + align - 1) / align * align;
    const VkDeviceSize off = uboSlots_.size() * stride;
    if (off + stride > kUboSlots * stride) throw std::runtime_error("out of uniform slots");
    Constants c{};
    c.hdr = opts_.hdr;
    c.depth = opts_.depth ? (opts_.depthReversed ? 2u : 1u) : 0u;
    c.timestamp = timestamp;
    std::memcpy(uboMap_ + off, &c, sizeof c);
    uboSlots_.emplace_back(timestamp, off);
    return off;
}

void Context::add(std::vector<Dispatch>& list, const std::string& shader, const Binds& b, VkExtent2D e, uint32_t div) {
    const Pipeline& p = dev_.pipeline(shader);
    if (b.b.size() != p.bindings.size())
        throw std::runtime_error(shader + ": graph binds " + std::to_string(b.b.size()) + " descriptors, shader declares " +
                                 std::to_string(p.bindings.size()));
    for (size_t i = 0; i < b.b.size(); ++i)
        if (b.b[i].kind != p.bindings[i].kind)
            throw std::runtime_error(shader + ": binding " + std::to_string(i) + " is a " +
                                     std::string(descriptorKindName(b.b[i].kind)) + " in the graph but a " +
                                     std::string(descriptorKindName(p.bindings[i].kind)) + " in the shader");
    list.push_back({&p, shader, b.b, (e.width + div - 1) / div, (e.height + div - 1) / div});
}

void Context::buildFlowGraph() {
    using S = SamplerKind;
    const uint32_t G = opts_.generated;
    const VkExtent2D full = opts_.extent;

    if (!dev_.nullDescriptor()) zero_ = makeImage({1, 1}, kRG32F);
    for (int i = 0; i < 2; ++i) inputs_[i] = *makeImage(full, frameFormat_, opts_.exportable);
    for (int i = 0; i < 2; ++i)
        depthInputs_[i] = *makeImage(opts_.depth ? full : VkExtent2D{1, 1}, kR32F, opts_.depth && opts_.exportable);
    for (uint32_t i = 0; i < G; ++i) outputs_.push_back(*makeImage(full, frameFormat_, opts_.exportable));

    const float l0Scale = opts_.flowScale * (dev_.performance() ? 0.25f : 0.5f);
    const VkExtent2D l0{std::max(32u, uint32_t(float(full.width) * l0Scale)),
                        std::max(32u, uint32_t(float(full.height) * l0Scale))};
    std::vector<VkExtent2D> levels;
    for (VkExtent2D e = l0;; e = halve(e)) {
        levels.push_back(e);
        if (e.width < 64 || e.height < 32 || levels.size() == 9) break;
    }
    const size_t L = levels.size();
    auto grid = [](VkExtent2D e) { return VkExtent2D{(e.width + 3) / 4, (e.height + 3) / 4}; };

    std::array<Group, 2> pyr;
    for (auto& p : pyr)
        for (const auto& e : levels) p.push_back(makeImage(e, kRG16F));
    Group raw, tmp, smooth;
    for (const auto& e : levels) {
        raw.push_back(makeImage(grid(e), kRG32F));
        tmp.push_back(makeImage(grid(e), kRG32F));
        smooth.push_back(makeImage(grid(e), kRG32F));
    }
    debugImages_.push_back({"input0", &inputs_[0]});
    debugImages_.push_back({"input1", &inputs_[1]});
    debugImages_.push_back({"luma0", pyr[0][0]});
    std::vector<Image*> flowOut(L);

    std::array<Image*, 2> still{makeImage(full, kR8), makeImage(full, kR8)};
    Image* refined = makeImage(levels[0], kRG32F);
    const bool refine = std::getenv("SYLUXIUM_MOTION_NO_REFINE") == nullptr;
    const VkDeviceSize ubo = uboSlot(0.5f);
    const size_t gainLevel = std::min<size_t>(4, L - 1);
    for (uint32_t f = 0; f < kPhases; ++f) {
        const uint32_t cur = f % 2, prev = (f + 1) % 2;
        auto& list = pre_[f];
        add(list, "flow.luma", Binds(*this).ubo(ubo).smp(S::Edge).tex(&inputs_[cur]).out(pyr[cur][0]), levels[0], 8);
        add(list, "flow.still",
            Binds(*this).ubo(ubo).smp(S::Edge).tex(&inputs_[prev]).tex(&inputs_[cur]).tex(still[prev]).out(still[cur]), full, 8);
        for (size_t l = 1; l < L; ++l)
            add(list, "flow.down", Binds(*this).smp(S::Edge).tex(pyr[cur][(l - 1) / 2 * 2]).out(pyr[cur][l]), levels[l], 8);
        const Image* coarse = nullptr;
        for (size_t l = L; l-- > 0;) {
            const int passes = l == 0 ? 2 : l == 1 ? 2 : l == 2 ? 1 : 0;
            Image* from = raw[l];
            Image* to = tmp[l];
            add(list, L >= 3 && l == L - 2 ? "flow.search.wide" : "flow.search", Binds(*this).smp(S::Edge).tex(pyr[prev][l]).tex(pyr[cur][l]).tex(coarse).tex(smooth[0]).out(from),
                grid(levels[l]), 8);
            for (int i = 0; i < passes; ++i) {
                add(list, "flow.prop", Binds(*this).smp(S::Edge).tex(pyr[prev][l]).tex(pyr[cur][l]).tex(from).out(to),
                    grid(levels[l]), 8);
                std::swap(from, to);
            }
            if (l == 0) {
                add(list, "flow.smooth", Binds(*this).smp(S::Edge).tex(from).out(smooth[l]), grid(levels[l]), 8);
                from = smooth[l];
            }
            coarse = from;
            flowOut[l] = from;
        }
        Image* interpFlow = flowOut[0];
        if (refine) {
            add(list, "flow.refine", Binds(*this).smp(S::Edge).tex(pyr[prev][0]).tex(pyr[cur][0]).tex(flowOut[0]).out(refined),
                levels[0], 8);
            interpFlow = refined;
        }
        main_[f].resize(G);
        for (uint32_t p = 0; p < G; ++p) {
            const float t = float(p + 1) / float(G + 1);
            const VkDeviceSize slot = uboSlot(t, true);
            timeSlots_[f].push_back(slot);
            add(main_[f][p], "flow.interp",
                Binds(*this).ubo(slot).smp(S::Edge).tex(&inputs_[prev]).tex(&inputs_[cur])
                    .tex(interpFlow).tex(pyr[prev][gainLevel]).tex(pyr[cur][gainLevel]).tex(still[cur]).tex(still[prev])
                    .tex(&depthInputs_[prev]).tex(&depthInputs_[cur]).out(&outputs_[p]), full, 8);
        }
    }
    for (size_t l = 0; l < L; ++l) debugImages_.push_back({"flow" + std::to_string(l), flowOut[l]});
    dispatchesPerFrame_ = pre_[0].size() + main_[0][0].size() * G;
}

void Context::allocateSets() {
    std::array<uint32_t, 4> counts{};
    uint32_t sets = 0;
    auto tally = [&](const std::vector<Dispatch>& l) {
        for (const auto& d : l) {
            ++sets;
            for (const auto& b : d.binds) ++counts[size_t(b.kind)];
        }
    };
    for (uint32_t f = 0; f < kPhases; ++f) {
        tally(pre_[f]);
        for (const auto& l : main_[f]) tally(l);
    }
    std::vector<VkDescriptorPoolSize> sizes;
    for (size_t k = 0; k < 4; ++k)
        if (counts[k]) sizes.push_back({vkType(DescriptorKind(k)), counts[k]});
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = sets;
    dpi.poolSizeCount = uint32_t(sizes.size());
    dpi.pPoolSizes = sizes.data();
    const auto& vk = dev_.vk();
    const VkDevice d = dev_.handle();
    check(vk.vkCreateDescriptorPool(d, &dpi, nullptr, &pool_), "descriptor pool");

    auto write = [&](std::vector<Dispatch>& l) {
        for (auto& disp : l) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &disp.pipeline->setLayout;
            check(vk.vkAllocateDescriptorSets(d, &ai, &disp.set), "descriptor set");

            std::vector<VkDescriptorImageInfo> images(disp.binds.size());
            std::vector<VkDescriptorBufferInfo> buffers(disp.binds.size());
            std::vector<VkWriteDescriptorSet> writes;
            for (size_t i = 0; i < disp.binds.size(); ++i) {
                const auto& b = disp.binds[i];
                VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = disp.set;
                w.dstBinding = disp.pipeline->bindings[i].binding;
                w.descriptorCount = 1;
                w.descriptorType = vkType(b.kind);
                if (b.kind == DescriptorKind::UniformBuffer) {
                    buffers[i] = {ubo_, b.uboOffset, sizeof(Constants)};
                    w.pBufferInfo = &buffers[i];
                } else {
                    images[i] = {b.sampler, b.view, VK_IMAGE_LAYOUT_GENERAL};
                    w.pImageInfo = &images[i];
                }
                writes.push_back(w);
            }
            vk.vkUpdateDescriptorSets(d, uint32_t(writes.size()), writes.data(), 0, nullptr);
        }
    };
    for (uint32_t f = 0; f < kPhases; ++f) {
        write(pre_[f]);
        for (auto& l : main_[f]) write(l);
    }
}

VkCommandBuffer Context::oneShotBegin() {
    const auto& vk = dev_.vk();
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmdPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    check(vk.vkAllocateCommandBuffers(dev_.handle(), &ai, &cb), "command buffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void Context::oneShotSubmit(VkCommandBuffer cb, const VkSemaphoreSubmitInfo* wait, const VkSemaphoreSubmitInfo* signal) {
    const auto& vk = dev_.vk();
    check(vk.vkEndCommandBuffer(cb), "end command buffer");
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.waitSemaphoreInfoCount = wait ? 1 : 0;
    si.pWaitSemaphoreInfos = wait;
    si.signalSemaphoreInfoCount = signal ? 1 : 0;
    si.pSignalSemaphoreInfos = signal;
    check(vk.vkResetFences(dev_.handle(), 1, &oneShotFence_), "reset fence");
    {
        std::lock_guard g(dev_.submitLock());
        check(vk.vkQueueSubmit2(dev_.queue(), 1, &si, oneShotFence_), "submit");
    }
    check(vk.vkWaitForFences(dev_.handle(), 1, &oneShotFence_, VK_TRUE, 5'000'000'000ull), "wait (one-shot)");
}

void Context::initLayouts() {
    std::vector<VkImageMemoryBarrier2> barriers;
    auto add = [&](const Image& i) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = i.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(b);
    };
    for (const auto& i : images_) add(*i);
    VkCommandBuffer cb = oneShotBegin();
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = uint32_t(barriers.size());
    dep.pImageMemoryBarriers = barriers.data();
    dev_.vk().vkCmdPipelineBarrier2(cb, &dep);
    const VkClearColorValue zero{};
    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (const auto& i : images_) dev_.vk().vkCmdClearColorImage(cb, i->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
    oneShotSubmit(cb, nullptr, nullptr);
}

void Context::record() {
    const auto& vk = dev_.vk();
    VkMemoryBarrier2 between{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    between.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    between.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    between.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    between.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo depBetween{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depBetween.memoryBarrierCount = 1;
    depBetween.pMemoryBarriers = &between;
    VkMemoryBarrier2 entry = between;
    entry.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    entry.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    entry.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo depEntry{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depEntry.memoryBarrierCount = 1;
    depEntry.pMemoryBarriers = &entry;
    VkMemoryBarrier2 exit = between;
    exit.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    exit.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo depExit{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depExit.memoryBarrierCount = 1;
    depExit.pMemoryBarriers = &exit;

    if (std::getenv("SYLUXIUM_MOTION_GPU_PROFILE") && queries_ && !prof_) {   
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 128;
        check(vk.vkCreateQueryPool(dev_.handle(), &qpi, nullptr, &prof_), "profile query pool");
    }
    bool profile = false;
    auto recordList = [&](VkCommandBuffer cb, const std::vector<Dispatch>& list) {
        vk.vkCmdPipelineBarrier2(cb, &depEntry);
        if (profile && profLabels_.empty()) {
            vk.vkCmdResetQueryPool(cb, prof_, 0, 128);
            vk.vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_, 0);
            profLabels_.push_back("start");
        }
        std::vector<VkImageView> written, read;
        auto has = [](const std::vector<VkImageView>& v, VkImageView x) { return std::find(v.begin(), v.end(), x) != v.end(); };
        for (const auto& d : list) {
            if (const char* skip = std::getenv("SYLUXIUM_MOTION_SKIP"))
                if ((d.label + " " + std::to_string(d.groupsX * 8) + "x").find(skip) != std::string::npos) continue;
            bool depends = false;
            for (const auto& b : d.binds) {
                if (!b.view) continue;
                if (b.kind == DescriptorKind::SampledImage) depends = depends || has(written, b.view);
                if (b.kind == DescriptorKind::StorageImage) depends = depends || has(written, b.view) || has(read, b.view);
            }
            if (depends) {
                vk.vkCmdPipelineBarrier2(cb, &depBetween);
                written.clear();
                read.clear();
            }
            for (const auto& b : d.binds) {
                if (!b.view) continue;
                if (b.kind == DescriptorKind::SampledImage) read.push_back(b.view);
                if (b.kind == DescriptorKind::StorageImage) written.push_back(b.view);
            }
            vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline->pipeline);
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline->layout, 0, 1, &d.set, 0, nullptr);
            vk.vkCmdDispatch(cb, d.groupsX, d.groupsY, 1);
            if (profile && profLabels_.size() < 128) {
                vk.vkCmdPipelineBarrier2(cb, &depBetween);
                vk.vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_, uint32_t(profLabels_.size()));
                profLabels_.push_back(d.label + " " + std::to_string(d.groupsX * 8) + "x" + std::to_string(d.groupsY * 8));
                written.clear();
                read.clear();
            }
        }
        vk.vkCmdPipelineBarrier2(cb, &depExit);
    };

    const uint32_t G = opts_.generated;
    std::vector<VkCommandBuffer> all(kPhases * (1 + G));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmdPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = uint32_t(all.size());
    check(vk.vkAllocateCommandBuffers(dev_.handle(), &ai, all.data()), "command buffers");

    size_t next = 0;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    for (uint32_t f = 0; f < kPhases; ++f) {
        VkCommandBuffer cb = preCmd_[f] = all[next++];
        vk.vkBeginCommandBuffer(cb, &bi);
        if (queries_) {
            vk.vkCmdResetQueryPool(cb, queries_, 2 * f, 2);
            vk.vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries_, 2 * f);
        }
        profile = prof_ && f == 0;
        recordList(cb, pre_[f]);
        check(vk.vkEndCommandBuffer(cb), "record pre-pass");

        mainCmd_[f].resize(G);
        for (uint32_t p = 0; p < G; ++p) {
            VkCommandBuffer mc = mainCmd_[f][p] = all[next++];
            vk.vkBeginCommandBuffer(mc, &bi);
            profile = prof_ && f == 0 && p == 0;
            recordList(mc, main_[f][p]);
            if (queries_ && p == G - 1)
                vk.vkCmdWriteTimestamp2(mc, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queries_, 2 * f + 1);
            check(vk.vkEndCommandBuffer(mc), "record main pass");
        }
    }
}

void Context::submit(uint64_t k) { submit(k, opts_.generated, nullptr); }

void Context::submit(uint64_t k, uint32_t count, const float* times) {
    if (k != submitted_) throw std::runtime_error("frames must be submitted in order");
    const auto& vk = dev_.vk();
    const uint32_t f = uint32_t(k % kPhases);
    const uint32_t G = opts_.generated;
    if (count > G) throw std::runtime_error("more generated frames requested than the context has");
    readTiming(k);

    if (times && count) {
        if (k >= kPhases) {
            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            const uint64_t v = generated(k - kPhases, G - 1);
            wi.semaphoreCount = 1;
            wi.pSemaphores = &timeline_;
            wi.pValues = &v;
            check(vk.vkWaitSemaphores(dev_.handle(), &wi, 1'000'000'000ull), "wait for uniform slot");
        }
        for (uint32_t p = 0; p < count; ++p) {
            const float t = std::clamp(times[p], 0.0f, 1.0f);
            std::memcpy(uboMap_ + timeSlots_[f][p] + offsetof(Constants, timestamp), &t, sizeof t);
        }
    }

    std::vector<VkCommandBufferSubmitInfo> cbs(1 + count, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO});
    std::vector<VkSemaphoreSubmitInfo> signals(std::max(count, 1u), {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO});
    std::vector<VkSubmitInfo2> infos(1 + count, {VK_STRUCTURE_TYPE_SUBMIT_INFO_2});

    VkSemaphoreSubmitInfo waitInput{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitInput.semaphore = timeline_;
    waitInput.value = inputReady(k);
    waitInput.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    VkSemaphoreSubmitInfo waitConsumed{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitConsumed.semaphore = consumed_;
    waitConsumed.value = k;
    waitConsumed.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    cbs[0].commandBuffer = preCmd_[f];
    infos[0].waitSemaphoreInfoCount = 1;
    infos[0].pWaitSemaphoreInfos = &waitInput;
    infos[0].commandBufferInfoCount = 1;
    infos[0].pCommandBufferInfos = &cbs[0];
    if (count == 0) {
        signals[0].semaphore = timeline_;
        signals[0].value = generated(k, G - 1);
        signals[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        infos[0].signalSemaphoreInfoCount = 1;
        infos[0].pSignalSemaphoreInfos = &signals[0];
    }
    for (uint32_t p = 0; p < count; ++p) {
        cbs[1 + p].commandBuffer = mainCmd_[f][p];
        signals[p].semaphore = timeline_;
        signals[p].value = generated(k, p + 1 == count ? G - 1 : p);
        signals[p].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        auto& si = infos[1 + p];
        if (p == 0) {
            si.waitSemaphoreInfoCount = 1;
            si.pWaitSemaphoreInfos = &waitConsumed;
        }
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbs[1 + p];
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &signals[p];
    }
    {
        std::lock_guard g(dev_.submitLock());
        check(vk.vkQueueSubmit2(dev_.queue(), uint32_t(infos.size()), infos.data(), VK_NULL_HANDLE),
              "submit frame generation");
    }
    ++submitted_;
}

void Context::readTiming(uint64_t k) {
    if (!queries_ || k < kPhases - 1) return;
    const uint32_t slot = uint32_t((k + 1) % kPhases);
    uint64_t ts[2];
    if (dev_.vk().vkGetQueryPoolResults(dev_.handle(), queries_, 2 * slot, 2, sizeof ts, ts, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] > ts[0])
        lastGpuMs_ = double(ts[1] - ts[0]) * dev_.timestampPeriodNs() / 1e6;
}

namespace {
int exportMemory(const Device& dev, VkDeviceMemory m) {
    if (!dev.vk().vkGetMemoryFdKHR) throw std::runtime_error(dev.name() + " can't export memory");
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gi.memory = m;
    gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    check(dev.vk().vkGetMemoryFdKHR(dev.handle(), &gi, &fd), "export memory");
    return fd;
}
int exportSemaphore(const Device& dev, VkSemaphore s) {
    if (!dev.vk().vkGetSemaphoreFdKHR) throw std::runtime_error(dev.name() + " can't export semaphores");
    VkSemaphoreGetFdInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    gi.semaphore = s;
    gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    check(dev.vk().vkGetSemaphoreFdKHR(dev.handle(), &gi, &fd), "export semaphore");
    return fd;
}
}

int Context::exportInput(int i) const { return exportMemory(dev_, inputs_.at(size_t(i)).memory); }
int Context::exportDepthInput(int i) const { return exportMemory(dev_, depthInputs_.at(size_t(i)).memory); }

void Context::setDepthMode(uint32_t mode) {
    if (!opts_.depth) mode = 0;
    for (const auto& [t, off] : uboSlots_) {
        (void)t;
        std::memcpy(uboMap_ + off + offsetof(Constants, depth), &mode, sizeof mode);
    }
}
int Context::exportOutput(uint32_t i) const { return exportMemory(dev_, outputs_.at(i).memory); }
int Context::exportTimeline() const { return exportSemaphore(dev_, timeline_); }
int Context::exportConsumed() const { return exportSemaphore(dev_, consumed_); }

namespace {
struct Staging {
    const Device& dev;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
    Staging(const Device& d, VkDeviceSize size) : dev(d) {
        const auto& vk = dev.vk();
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vk.vkCreateBuffer(dev.handle(), &bci, nullptr, &buf), "staging buffer");
        VkMemoryRequirements mr;
        vk.vkGetBufferMemoryRequirements(dev.handle(), buf, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = dev.memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vk.vkAllocateMemory(dev.handle(), &mai, nullptr, &mem), "staging memory");
        vk.vkBindBufferMemory(dev.handle(), buf, mem, 0);
        vk.vkMapMemory(dev.handle(), mem, 0, VK_WHOLE_SIZE, 0, &map);
    }
    ~Staging() {
        dev.vk().vkDestroyBuffer(dev.handle(), buf, nullptr);
        dev.vk().vkFreeMemory(dev.handle(), mem, nullptr);
    }
};
}

void Context::testUpload(uint64_t k, const std::vector<uint8_t>& rgba, const std::vector<float>* depth) {
    const VkExtent2D e = opts_.extent;
    if (frameFormat_ != kRGBA8 || rgba.size() != size_t(e.width) * e.height * 4)
        throw std::runtime_error("testUpload expects tightly packed RGBA8 at the context size");
    if (k > 0) {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        const uint64_t v = generated(k - 1, opts_.generated - 1);
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timeline_;
        wi.pValues = &v;
        check(dev_.vk().vkWaitSemaphores(dev_.handle(), &wi, 5'000'000'000ull), "wait for previous frame");
    }
    Staging st(dev_, rgba.size());
    std::memcpy(st.map, rgba.data(), rgba.size());
    VkCommandBuffer cb = oneShotBegin();
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {e.width, e.height, 1};
    dev_.vk().vkCmdCopyBufferToImage(cb, st.buf, inputs_[k % 2].image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    std::unique_ptr<Staging> dst;
    if (opts_.depth && depth) {
        if (depth->size() != size_t(e.width) * e.height) throw std::runtime_error("testUpload: depth size mismatch");
        dst = std::make_unique<Staging>(dev_, depth->size() * sizeof(float));
        std::memcpy(dst->map, depth->data(), depth->size() * sizeof(float));
        dev_.vk().vkCmdCopyBufferToImage(cb, dst->buf, depthInputs_[k % 2].image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = timeline_;
    sig.value = inputReady(k);
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    oneShotSubmit(cb, nullptr, &sig);
}

std::vector<uint8_t> Context::testDownload(uint64_t k, uint32_t i) {
    const VkExtent2D e = opts_.extent;
    const size_t bytes = size_t(e.width) * e.height * 4;
    Staging st(dev_, bytes);
    VkCommandBuffer cb = oneShotBegin();
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {e.width, e.height, 1};
    dev_.vk().vkCmdCopyImageToBuffer(cb, outputs_.at(i).image, VK_IMAGE_LAYOUT_GENERAL, st.buf, 1, &region);
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = timeline_;
    wait.value = generated(k, i);
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    oneShotSubmit(cb, &wait, nullptr);
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), st.map, bytes);
    return out;
}

namespace {
float halfToFloat(uint16_t h);
}

namespace {
float halfToFloat(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}
}

std::vector<uint8_t> Context::debugDownload(const Image& img) {
    dev_.vk().vkQueueWaitIdle(dev_.queue());
    const bool half = img.format == kRGBA16F;
    const size_t px = size_t(img.extent.width) * img.extent.height;
    Staging st(dev_, px * (half ? 8 : 4));
    VkCommandBuffer cb = oneShotBegin();
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {img.extent.width, img.extent.height, 1};
    dev_.vk().vkCmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_GENERAL, st.buf, 1, &region);
    oneShotSubmit(cb, nullptr, nullptr);
    std::vector<uint8_t> out(px * 4);
    if (img.format == kR32F) {
        const float* f = static_cast<const float*>(st.map);
        float lo = 1e30f, hi = -1e30f;
        for (size_t i = 0; i < px; ++i) { lo = std::min(lo, f[i]); hi = std::max(hi, f[i]); }
        for (size_t i = 0; i < px; ++i) {
            const uint8_t g = uint8_t(hi > lo ? (f[i] - lo) / (hi - lo) * 255.0f + 0.5f : 0.0f);
            out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = g;
            out[4 * i + 3] = 255;
        }
        return out;
    }
    for (size_t i = 0; i < px * 4; ++i) {
        if (half) {
            const float v = halfToFloat(static_cast<const uint16_t*>(st.map)[i]);
            out[i] = uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
        } else {
            out[i] = static_cast<const uint8_t*>(st.map)[i];
        }
    }
    return out;
}

std::vector<std::string> Context::debugStats() {
    std::vector<std::string> out;
    dev_.vk().vkQueueWaitIdle(dev_.queue());
    if (prof_) {
        std::vector<uint64_t> ts(profLabels_.size());
        if (dev_.vk().vkGetQueryPoolResults(dev_.handle(), prof_, 0, uint32_t(ts.size()), ts.size() * 8, ts.data(), 8,
                                            VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            for (size_t i = 1; i < ts.size(); ++i) {
                char l[160];
                std::snprintf(l, sizeof l, "profile %-28s %7.1f us", profLabels_[i].c_str(),
                              double(ts[i] - ts[i - 1]) * dev_.timestampPeriodNs() / 1e3);
                out.push_back(l);
            }
    }
    for (const auto& [name, img] : debugImages_) {
        if (img->format == kRG16F) continue;
        if (img->format == kRG32F || img->format == kR32F) {
            const uint32_t ch = img->format == kRG32F ? 2 : 1;
            const size_t px = size_t(img->extent.width) * img->extent.height;
            Staging st(dev_, px * ch * 4);
            VkCommandBuffer cb = oneShotBegin();
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {img->extent.width, img->extent.height, 1};
            dev_.vk().vkCmdCopyImageToBuffer(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, st.buf, 1, &region);
            oneShotSubmit(cb, nullptr, nullptr);
            const float* f = static_cast<const float*>(st.map);
            char line[512];
            if (const char* dump = ch == 2 ? std::getenv("SYLUXIUM_MOTION_FLOW_DUMP") : nullptr) {
                if (FILE* fp = std::fopen((std::string(dump) + "/" + name + ".bin").c_str(), "wb")) {
                    std::fwrite(&img->extent.width, 4, 1, fp);
                    std::fwrite(&img->extent.height, 4, 1, fp);
                    std::fwrite(f, 4, px * 2, fp);
                    std::fclose(fp);
                }
            }
            if (ch == 2) {
                std::map<int, size_t> hist;
                double sx = 0, sy = 0;
                for (size_t i = 0; i < px; ++i) {
                    const double vx = f[2 * i] * opts_.extent.width, vy = f[2 * i + 1] * opts_.extent.height;
                    sx += vx; sy += vy;
                    ++hist[int(std::lround(vx))];
                }
                std::vector<std::pair<size_t, int>> top;
                for (auto [v, n] : hist) top.push_back({n, v});
                std::sort(top.rbegin(), top.rend());
                int n = std::snprintf(line, sizeof line, "%-18s %4ux%-4u mean (%.2f, %.2f) px; most common x:", name.c_str(),
                                      img->extent.width, img->extent.height, sx / double(px), sy / double(px));
                for (size_t i = 0; i < std::min<size_t>(4, top.size()); ++i)
                    n += std::snprintf(line + n, sizeof line - size_t(n), " %d px (%.0f%%)", top[i].second,
                                       100.0 * double(top[i].first) / double(px));
            } else {
                double mn = 1e30, mx = -1e30, sum = 0;
                for (size_t i = 0; i < px; ++i) { mn = std::min(mn, double(f[i])); mx = std::max(mx, double(f[i])); sum += f[i]; }
                std::snprintf(line, sizeof line, "%-18s %4ux%-4u [%.3f..%.3f avg %.3f]", name.c_str(), img->extent.width,
                              img->extent.height, mn, mx, sum / double(px));
            }
            out.push_back(line);
            continue;
        }
        const uint32_t ch = img->format == kR8 ? 1 : 4;
        const uint32_t bpc = img->format == kRGBA16F ? 2 : 1;
        const size_t bytes = size_t(img->extent.width) * img->extent.height * ch * bpc;
        Staging st(dev_, bytes);
        VkCommandBuffer cb = oneShotBegin();
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {img->extent.width, img->extent.height, 1};
        dev_.vk().vkCmdCopyImageToBuffer(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, st.buf, 1, &region);
        oneShotSubmit(cb, nullptr, nullptr);
        double mn[4] = {1e30, 1e30, 1e30, 1e30}, mx[4] = {-1e30, -1e30, -1e30, -1e30}, sum[4] = {};
        const size_t px = size_t(img->extent.width) * img->extent.height;
        for (size_t i = 0; i < px; ++i)
            for (uint32_t c = 0; c < ch; ++c) {
                double v = bpc == 2 ? halfToFloat(static_cast<const uint16_t*>(st.map)[i * ch + c])
                                    : static_cast<const uint8_t*>(st.map)[i * ch + c] / 255.0;
                mn[c] = std::min(mn[c], v);
                mx[c] = std::max(mx[c], v);
                sum[c] += v;
            }
        char line[512];
        int n = std::snprintf(line, sizeof line, "%-18s %4ux%-4u", name.c_str(), img->extent.width, img->extent.height);
        for (uint32_t c = 0; c < ch; ++c)
            n += std::snprintf(line + n, sizeof line - size_t(n), "  [%.3f..%.3f avg %.3f]", mn[c], mx[c], sum[c] / double(px));
        out.push_back(line);
    }
    return out;
}

void Context::testConsumed(uint64_t k) {
    VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    si.semaphore = consumed_;
    si.value = k + 1;
    check(dev_.vk().vkSignalSemaphore(dev_.handle(), &si), "signal consumed");
}

}
