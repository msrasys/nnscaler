// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include <torch/extension.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

cudaError_t launch_activation_sm_copy(void*, const void*, int64_t, int, cudaStream_t);

void copy_h2d(torch::Tensor destination, torch::Tensor source, int blocks) {
    TORCH_CHECK(destination.is_cuda() && source.device().is_cpu(), "SM restore requires CPU -> CUDA");
    TORCH_CHECK(destination.scalar_type() == at::kByte && source.scalar_type() == at::kByte,
                "SM restore copies raw uint8 storage only");
    TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(), "SM restore requires contiguous storage");
    TORCH_CHECK(destination.numel() == source.numel(), "SM restore sizes must match");
    TORCH_CHECK(blocks > 0, "SM restore block count must be positive");
    if (!source.numel()) return;
    c10::cuda::CUDAGuard guard(destination.device());
    TORCH_CHECK(source.is_pinned(), "SM restore requires pinned host memory");
    const auto stream = c10::cuda::getCurrentCUDAStream(destination.get_device());
    void* mapped_base = nullptr;
    auto* host_base = source.storage().data_ptr().get();
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_base, host_base, 0));
    auto* mapped = static_cast<char*>(mapped_base)
                 + (static_cast<char*>(source.data_ptr()) - static_cast<char*>(host_base));
    // Keep host allocations from being recycled while the kernel reads them.
    // This compatibility wrapper forwards to the unified allocator on newer PyTorch.
    TORCH_CHECK(at::cuda::CachingHostAllocator_recordEvent(
                    source.data_ptr(), source.storage().data_ptr().get_context(), stream),
                "SM restore requires a PyTorch-managed pinned allocation");
    c10::cuda::CUDACachingAllocator::recordStream(destination.storage().data_ptr(), stream);
    C10_CUDA_CHECK(launch_activation_sm_copy(destination.data_ptr(), mapped, source.numel(), blocks, stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("copy_h2d", &copy_h2d, pybind11::call_guard<pybind11::gil_scoped_release>());
}
