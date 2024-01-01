/*
 * Copyright (c) 2023, Tri Dao.
 */
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "cutlass/numeric_types.h"
#include "cutlass/integer_subbyte.h"

#include "cutlass_extensions/weight_only_quant_op.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

#include "static_switch.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define QUANTOP_SWITCH(QUANTOP, CONST_NAME, ...)                                                        \
    [&] {                                                                                               \
        if (QUANTOP == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY) {                             \
            static constexpr auto CONST_NAME = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;       \
            return __VA_ARGS__();                                                                       \
        } else if (QUANTOP == cutlass::WeightOnlyQuantOp::PER_TENSOR_ONLY) {                            \
            static constexpr auto CONST_NAME = cutlass::WeightOnlyQuantOp::PER_TENSOR_ONLY;             \
            return __VA_ARGS__();                                                                       \
        } else if (QUANTOP == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY) {                     \
            static constexpr auto CONST_NAME = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;      \
            return __VA_ARGS__();                                                                       \
        } else {                                                                                        \
            static constexpr auto CONST_NAME = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS; \
            return __VA_ARGS__();                                                                       \
        }                                                                                               \
    }()

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void dispatch_to_weight_only_batched_gemv(const half* A, const WeightType* B, const half* weight_scales,
    const half* weight_zero_points, const half* bias, half* C, int m, int n, int k, int group_size, float global_scale, cudaStream_t stream)
{
    using namespace tensorrt_llm::kernels;

    // Convert WeightType
    WeightOnlyQuantType weight_only_quant_type
        = std::is_same_v<WeightType, cutlass::uint4b_t> ? WeightOnlyQuantType::Int4b : WeightOnlyQuantType::Int8b;

    // Convert QuantType
    WeightOnlyType weight_only_type =
        (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY || QuantOp == cutlass::WeightOnlyQuantOp::PER_TENSOR_ONLY)
        ? WeightOnlyType::PerChannel
        : WeightOnlyType::GroupWise;

    // https://github.com/NVIDIA/TensorRT-LLM/blob/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.cpp#L322
    // https://github.com/NVIDIA/TensorRT-LLM/blob/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.cpp#L363
    WeightOnlyParams params{
        /*qweight=*/reinterpret_cast<const uint8_t*>(B),
        /*scales=*/reinterpret_cast<const half*>(weight_scales),
        /*zeros=*/reinterpret_cast<const half*>(weight_zero_points),
        /*in=*/reinterpret_cast<const half*>(A),
        /*act_scale=*/nullptr,
        /*bias=*/reinterpret_cast<const half*>(bias),
        /*out=*/reinterpret_cast<half*>(C),
        m,
        n,
        k,
        group_size,
        global_scale,
        weight_only_quant_type,
        weight_only_type,
        WeightOnlyActivationFunctionType::Identity,
        WeightOnlyActivationType::FP16
    };

    weight_only_batched_gemv_launcher(params, stream);
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void gemm_fp16_int_bias(const half* A, const WeightType* B, const half* weight_scales, const half* weight_zero_points,
    const half* bias, half* C, int m, int n, int k, int group_size, float global_scale,
    char* workspace_ptr, size_t workspace_bytes, cudaStream_t stream)
{
    if (m <= 4) {
        dispatch_to_weight_only_batched_gemv<WeightType, QuantOp>(A, B, weight_scales, weight_zero_points,
            bias, C, m, n, k, group_size, global_scale, stream);
    } else {
        using namespace tensorrt_llm::kernels::cutlass_kernels;
        CutlassFpAIntBGemmRunner<half, WeightType, QuantOp> runner;
        runner.gemm_bias(A, B, weight_scales, weight_zero_points, bias, C, m, n, k, group_size, global_scale, workspace_ptr, workspace_bytes, stream);
    }

}

at::Tensor preprocess_weight(at::Tensor quantized_weight, int bits) {

    TORCH_CHECK(bits == 4 || bits == 8);
    int rows = quantized_weight.size(0);
    int elts_per_byte = 8 / bits;
    int cols = quantized_weight.size(1) * elts_per_byte;

    TORCH_CHECK(quantized_weight.dtype() == torch::kInt8);
    TORCH_CHECK(quantized_weight.is_cpu());
    TORCH_CHECK(quantized_weight.is_contiguous());
    CHECK_SHAPE(quantized_weight, rows, cols / elts_per_byte);

    auto opts = quantized_weight.options();
    auto out = at::empty({cols, rows / elts_per_byte}, opts);

    using namespace tensorrt_llm::kernels::cutlass_kernels;
    QuantType qtype = bits == 4 ? QuantType::PACKED_INT4_WEIGHT_ONLY : QuantType::INT8_WEIGHT_ONLY;
    std::vector<size_t> shape{rows, cols};
    preprocess_weights_for_mixed_gemm(out.data_ptr<int8_t>(), quantized_weight.data_ptr<int8_t>(), shape, qtype);
    return out;
}

at::Tensor quant_matmul(const at::Tensor input, const at::Tensor weight,
                        const c10::optional<at::Tensor> weight_scales_,
                        const c10::optional<at::Tensor> weight_zero_points_, float global_scale,
                        const c10::optional<at::Tensor> bias_, int bits) {

    TORCH_CHECK(bits == 4 || bits == 8);
    const int m = input.size(0);
    const int k = input.size(1);
    const int n = weight.size(0);
    TORCH_CHECK(n % 8 == 0);

    TORCH_CHECK(input.dtype() == torch::kFloat16);
    TORCH_CHECK(weight.dtype() == torch::kInt8);
    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(weight.is_cuda());
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(weight.is_contiguous());
    CHECK_SHAPE(input, m, k);
    CHECK_SHAPE(weight, n, k / (8 / bits));

    bool is_finegrained = false;
    int group_size = k;
    if (weight_scales_.has_value()) {
        auto weight_scales = weight_scales_.value();
        TORCH_CHECK(weight_scales.dtype() == torch::kFloat16);
        TORCH_CHECK(weight_scales.is_cuda());
        TORCH_CHECK(weight_scales.is_contiguous());
        is_finegrained = weight_scales.dim() == 2;
        group_size = !is_finegrained ? k : k / weight_scales.size(0);
        TORCH_CHECK(k % group_size == 0);
        if (is_finegrained) { TORCH_CHECK(group_size == 64 || group_size == 128, "Only support group size 64 and 128"); }
        if (!is_finegrained) {
            CHECK_SHAPE(weight_scales, n);
        } else {
            CHECK_SHAPE(weight_scales, k / group_size, n);
        }
    } else {
        TORCH_CHECK(!weight_zero_points_.has_value(), "If weight_scales is None, weight_zero_points must also be None");
    }

    if (weight_zero_points_.has_value()) {
        TORCH_CHECK(is_finegrained, "weight_zero_points is only supported if using groupwise quantization");
        auto weight_zero_points = weight_zero_points_.value();
        TORCH_CHECK(weight_zero_points.dtype() == torch::kFloat16);
        TORCH_CHECK(weight_zero_points.is_cuda());
        TORCH_CHECK(weight_zero_points.is_contiguous());
        CHECK_SHAPE(weight_zero_points, k / group_size, n);
    }

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.dtype() == torch::kFloat16);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.is_contiguous());
        CHECK_SHAPE(bias, n);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};

    // create output/workspace tensor
    auto opts = input.options();
    auto out = at::empty({m, n}, opts);
    at::Tensor workspace;
    bool has_workspace = m > 4;  // m <= 4 dispatches to batched gemv which doesn't need workspace.
    if (has_workspace) { workspace = at::empty({1 << 22}, opts.dtype(torch::kInt8)); }

    auto quantop = !is_finegrained
        ? (weight_scales_.has_value() ? cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY : cutlass::WeightOnlyQuantOp::PER_TENSOR_ONLY)
        : (!weight_zero_points_.has_value() ? cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY : cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
    BOOL_SWITCH(bits == 4, kIs4Bits, [&] {
        using WeightType = std::conditional_t<kIs4Bits, cutlass::uint4b_t, uint8_t>;
        QUANTOP_SWITCH(quantop, kQuantOp, [&] {
            gemm_fp16_int_bias<WeightType, kQuantOp>(
                reinterpret_cast<half *>(input.data_ptr()),
                reinterpret_cast<WeightType *>(weight.data_ptr()),
                weight_scales_.has_value()? reinterpret_cast<half *>(weight_scales_.value().data_ptr()) : nullptr,
                weight_zero_points_.has_value() ? reinterpret_cast<half *>(weight_zero_points_.value().data_ptr()) : nullptr,
                bias_.has_value() ? reinterpret_cast<half *>(bias_.value().data_ptr()) : nullptr,
                reinterpret_cast<half *>(out.data_ptr()),
                m,
                n,
                k,
                group_size,
                global_scale,
                has_workspace ? reinterpret_cast<char *>(workspace.data_ptr()) : nullptr,
                has_workspace ? 1 << 22 : 0,
                at::cuda::getCurrentCUDAStream());
        });
    });

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_weight", &preprocess_weight, "Preprocess weight");
  m.def("quant_matmul", &quant_matmul, "Quant matmul");
}
