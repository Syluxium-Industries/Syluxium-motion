# Syluxium Motion

> **Early beta.** Syluxium Motion is at an early stage: expect rough edges, changes to the API between
> versions, and little documentation beyond this page. It is an engine, not an application. It doesn't
> capture frames from games, decide when frames are shown or present them on screen, and has no user
> interface: you build your own program around it to get frames in and out. Use it at your own risk;
> issues and feedback are welcome.

GPU frame generation on Vulkan: it estimates motion between two real frames and renders in-between
frames at any time between them. It is the engine behind LiquidFrames. MIT License, Copyright (c) 2026
Syluxium Industries.

The engine only generates frames. Getting frames from an application, deciding when to show which
frame (pacing) and presenting them are up to the program that uses it.

## How it works

- **Per real frame:** a luma + colour-difference image pyramid (Gaussian filtered), coarse-to-fine
  block matching for symmetric motion, propagation with neighbour consensus, a vector median, and
  detection of still overlays (HUD/UI unchanged for 3+ frames is left alone).
- **Per-pixel motion edges:** at the finest level every pixel picks, from the 3x3 blocks around it,
  the motion that matches its own 5x5 neighbourhood clearly better than the smooth field, so motion
  boundaries follow the picture's edges (occluded pixels, which match nothing, keep the smooth field).
- **Optional depth** (`ContextOptions::depth`, the application's depth buffer per real frame): where a
  pixel's own depth changes between the two frames (it is uncovered or covered), a motion whose two
  samples land on different surfaces shows the farther one: what's behind the object.
- **Per generated frame:** interpolation at any time t in (0, 1], with motion corrected for t,
  occlusion handling at moving-object edges, and brightness-matched candidate comparison.
- Command buffers are pre-recorded; one timeline semaphore orders everything.

Measured on an RX 7700 XT (sustained): about 1.3 ms per real frame at 720p, 2.0 ms at 1080p and
3.2 ms at 1440p with one generated frame; each extra generated frame adds about 0.25-0.45 ms. Performance
mode: about 1.35 ms at 1080p. Weak spots: very fast motion and thin
repeating patterns (fences).

## Requirements

- Vulkan 1.2 or newer with timeline semaphores, external memory/semaphore fds (for sharing frames) and
  storage image writes without format. synchronization2 is used when the driver has it (1.3 or the
  extension) and translated to the classic calls otherwise; robustness2 null descriptors are used when
  available (an all-zero image otherwise). That should also cover older drivers such as NVIDIA's 470 series for
  Kepler (GTX 600/700), not yet tested on such cards. `SYLUXIUM_MOTION_COMPAT=1` forces the bare 1.2
  path for testing (identical output). Tested on AMD with Mesa RADV; other drivers should work but are
  untested.
- Build: C++20 compiler, CMake 3.22+, `glslc`, the Vulkan headers. Vulkan itself is loaded at run time
  (`dlopen`), so nothing links libvulkan.

## Building

Standalone: `cmake -S . -B build -G Ninja && cmake --build build` makes `libsyluxium-motion.a`.
In another project: `add_subdirectory(path/to/motion)` and link `SyluxiumMotion::Motion`. Include
`syluxium_motion/device.hpp` and `syluxium_motion/context.hpp`. Namespace: `syluxium::motion`.

## Using it

```cpp
using namespace syluxium::motion;
Device dev({.gpuUuid = "", .interop = false});          // its own VkDevice ("" = first suitable GPU)
Context ctx(dev, {.extent = {1920, 1080}, .generated = 1});   // 1 generated frame per real frame (2x)

// For each real frame k = 0, 1, 2, ...
//  1. Write frame k into ctx.input(k % 2) (VK_IMAGE_LAYOUT_GENERAL, ctx.frameFormat()), after the
//     work for frame k-1 has finished (timeline >= ctx.generated(k - 1, generated - 1)).
//  2. Signal ctx.timelineSemaphore() to ctx.inputReady(k).
//  3. ctx.submit(k);                       // evenly spaced frames, or
//     ctx.submit(k, count, times);         // `count` frames at times in (0, 1] (adaptive pacing)
//  4. Output i of frame k, ctx.output(i), is ready when the timeline reaches ctx.generated(k, i).
//     It shows the scene between frame k-1 (t = 0) and frame k (t = 1).
//  5. After copying the outputs out, signal ctx.consumedSemaphore() to k + 1.
```

- `DeviceOptions`:
  - `gpuUuid`: pick a GPU.
  - `performance`: motion at quarter size, faster.
  - `interop`: share frames with another VkDevice through external memory/semaphore fds.
  - `hostImport`: frames arrive through host memory from another GPU.
  - `present`: the device will also present.
  - `disableLayerEnv`: an environment variable set to 1 while the engine creates its instance.
- `ContextOptions`:
  - `extent`
  - `hdr`: RGBA16F frames.
  - `flowScale`: 0.25-1, lower is faster.
  - `generated`: 1-7.
  - `exportable`: the frames and semaphores can be exported with `exportInput`, `exportOutput`,
    `exportTimeline` and `exportConsumed`, so another VkDevice in the same process (for example an
    application's) can write the inputs and read the outputs without copies.
  - `frameFormat`
  - `depth`, `depthReversed`: depth for frame k goes into `depthInput(k % 2)` (R32F, the depth
    buffer's 0..1 values; reversed = 1 is near) before `inputReady(k)`; `exportDepthInput(i)` shares it.
- `testUpload`, `testDownload` and `testConsumed` do the same steps through host memory (RGBA8), for
  tests and simple integrations.
- Debug environment variables: `SYLUXIUM_MOTION_GPU_PROFILE=1` (per-pass GPU times),
  `SYLUXIUM_MOTION_NO_REFINE=1` (skip the per-pixel motion pass, for comparisons),
  `SYLUXIUM_MOTION_SKIP`, `SYLUXIUM_MOTION_FLOW_DUMP=dir`.

## License

MIT License, see [LICENSE](LICENSE). Copyright (c) 2026 Syluxium Industries.
