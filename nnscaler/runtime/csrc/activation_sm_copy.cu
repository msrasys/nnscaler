// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

__global__ void activation_sm_copy(unsigned char* dst, const unsigned char* src, int64_t bytes, bool aligned) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    if (aligned) {
        const int64_t vectors = bytes / 16;
        auto* d = reinterpret_cast<uint4*>(dst);
        auto* s = reinterpret_cast<const uint4*>(src);
        for (int64_t i = index; i < vectors; i += stride) d[i] = s[i];
        for (int64_t i = vectors * 16 + index; i < bytes; i += stride) dst[i] = src[i];
    } else {
        for (int64_t i = index; i < bytes; i += stride) dst[i] = src[i];
    }
}

cudaError_t launch_activation_sm_copy(void* dst, const void* src, int64_t bytes, int blocks, cudaStream_t stream) {
    const bool aligned = ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0;
    const int64_t elements = aligned ? (bytes + 15) / 16 : bytes;
    blocks = int(std::min<int64_t>(blocks, (elements + 255) / 256));
    activation_sm_copy<<<blocks, 256, 0, stream>>>(static_cast<unsigned char*>(dst),
                                               static_cast<const unsigned char*>(src), bytes, aligned);
    return cudaGetLastError();
}
