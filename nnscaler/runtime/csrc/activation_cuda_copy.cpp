// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

cudaError_t launch_activation_sm_copy(void*, const void*, int64_t, int, cudaStream_t);

template<bool H2D>
void copy_bytes(torch::Tensor destination, torch::Tensor source, int blocks) {
    const auto& host = H2D ? source : destination;
    const auto& device = H2D ? destination : source;
    TORCH_CHECK(device.is_cuda() && host.device().is_cpu(),
                H2D ? "SM copy requires CPU -> CUDA" : "SM copy requires CUDA -> CPU");
    TORCH_CHECK(destination.scalar_type() == at::kByte && source.scalar_type() == at::kByte,
                "SM copy transfers raw uint8 storage only");
    TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(), "SM copy requires contiguous storage");
    TORCH_CHECK(destination.numel() == source.numel(), "SM copy sizes must match");
    TORCH_CHECK(blocks > 0, "SM copy block count must be positive");
    if (!source.numel()) return;
    c10::cuda::CUDAGuard guard(device.device());
    TORCH_CHECK(host.is_pinned(), "SM copy requires pinned host memory");
    const auto stream = c10::cuda::getCurrentCUDAStream(device.get_device());
    void* mapped_base = nullptr;
    auto* host_base = host.storage().data_ptr().get();
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_base, host_base, 0));
    auto* mapped = static_cast<char*>(mapped_base)
                 + (static_cast<char*>(host.data_ptr()) - static_cast<char*>(host_base));
    // Keep both allocations alive while the kernel reads or writes them.
    // This compatibility wrapper forwards to the unified allocator on newer PyTorch.
    TORCH_CHECK(at::cuda::CachingHostAllocator_recordEvent(
                    host.data_ptr(), host.storage().data_ptr().get_context(), stream),
                "SM copy requires a PyTorch-managed pinned allocation");
    c10::cuda::CUDACachingAllocator::recordStream(device.storage().data_ptr(), stream);
    C10_CUDA_CHECK(launch_activation_sm_copy(H2D ? device.data_ptr() : mapped,
                                           H2D ? mapped : device.data_ptr(),
                                           source.numel(), blocks, stream));
}

int64_t copy_dma(torch::Tensor destination, torch::Tensor source, int64_t chunk_bytes) {
    const bool h2d = destination.is_cuda();
    const auto& device = h2d ? destination : source;
    const auto& host = h2d ? source : destination;
    TORCH_CHECK(device.is_cuda() && host.device().is_cpu(), "Native DMA requires CPU <-> CUDA");
    TORCH_CHECK(destination.scalar_type() == at::kByte && source.scalar_type() == at::kByte,
                "Native DMA transfers raw uint8 storage only");
    TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(), "Native DMA requires contiguous storage");
    TORCH_CHECK(destination.numel() == source.numel(), "Native DMA sizes must match");
    TORCH_CHECK(chunk_bytes >= 0, "Native DMA chunk size must be non-negative");
    const int64_t bytes = source.numel();
    if (!bytes) return 0;
    c10::cuda::CUDAGuard guard(device.device());
    TORCH_CHECK(host.is_pinned(), "Native DMA requires pinned host memory");
    const auto stream = c10::cuda::getCurrentCUDAStream(device.get_device());
    // Record both allocations before submitting any copy. Even a partial
    // submission failure must not let either allocation be recycled early.
    TORCH_CHECK(at::cuda::CachingHostAllocator_recordEvent(
                    host.data_ptr(), host.storage().data_ptr().get_context(), stream),
                "Native DMA requires a PyTorch-managed pinned allocation");
    c10::cuda::CUDACachingAllocator::recordStream(device.storage().data_ptr(), stream);
    const int64_t chunk = chunk_bytes ? chunk_bytes : bytes;
    const auto kind = h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    auto* dst = static_cast<char*>(destination.data_ptr());
    const auto* src = static_cast<const char*>(source.data_ptr());
    int64_t copies = 0;
    for (int64_t offset = 0; offset < bytes;) {
        const int64_t count = std::min(chunk, bytes - offset);
        C10_CUDA_CHECK(cudaMemcpyAsync(dst + offset, src + offset,
                                      count, kind, stream));
        offset += count;
        ++copies;
    }
    return copies;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("copy_dma", &copy_dma, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("copy_h2d", &copy_bytes<true>, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("copy_d2h", &copy_bytes<false>, pybind11::call_guard<pybind11::gil_scoped_release>());
}
