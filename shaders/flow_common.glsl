// Copyright (c) 2026 Syluxium Industries.
// SPDX-License-Identifier: MIT
// Syluxium Motion: open source under the MIT License, see LICENSE.

layout(std140, binding = 0) uniform Constants {
    uint hdr;
    float timestamp;
    uint depthMode;
    float pad;
} C;

float depthGap(float a, float b) {
    const float ca = C.depthMode == 2u ? a : 1.0 - a, cb = C.depthMode == 2u ? b : 1.0 - b;
    return abs(ca - cb) / max(max(ca, cb), 1e-4);
}

float luma(vec3 c) {
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return C.hdr != 0u ? l / (1.0 + l) : l;
}
