// From
// https://github.com/tlc-pack/cutlass_fpA_intB_gemm/blob/main/cutlass_kernels/moe_gemm/moe_gemv_kernels.h

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"

namespace tensorrt_llm
{
namespace kernels
{

void moe_gemv(const half* A, const half* B, half* C, int64_t* total_rows_before_expert, int64_t total_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
