# Copyright (c) 2023, Tri Dao.

import math
import pytest

import torch

from quant_matmul import preprocess_weight, quant_matmul_fn
from quant_matmul.quant_matmul_interface import quant_matmul_ref

import quant_matmul_cuda


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("out_features", [64, 192, 2048, 2752, 4096, 5120])
@pytest.mark.parametrize("in_features", [192, 2048, 2560, 4096, 5120])
def test_preprocess_weight(in_features, out_features, bits):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    w = torch.randint(-128, 127, (out_features // (8 // bits), in_features), dtype=torch.int8, device=device)
    w_processed = preprocess_weight(w, bits)
    wpacked_ref = quant_matmul_cuda.preprocess_weight(w.T.cpu().contiguous(), bits).to(device)
    assert torch.equal(w_processed, wpacked_ref)


@pytest.mark.parametrize("bits", [8, 4])
# @pytest.mark.parametrize("bits", [8])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 16, 37])
# @pytest.mark.parametrize("batch", [1, 2, 3, 4])
# @pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("has_global_scale", [False, True])
# @pytest.mark.parametrize("has_global_scale", [True])
@pytest.mark.parametrize("has_zero_points", [False, True])
# @pytest.mark.parametrize("has_zero_points", [False])
@pytest.mark.parametrize("groupsize", [None, 64, 128])
# @pytest.mark.parametrize("groupsize", [None])
@pytest.mark.parametrize("out_features", [64, 192, 2048, 2752, 4096, 5120])
# @pytest.mark.parametrize("out_features", [64])
@pytest.mark.parametrize("in_features", [192, 2048, 2560, 4096, 5120])
# @pytest.mark.parametrize("in_features", [192])
def test_multiply(in_features, out_features, groupsize, has_zero_points, has_global_scale, batch, has_bias, bits):
    if groupsize is not None and in_features % groupsize != 0:
        pytest.skip("groupsize must be divisible by in_features")
    if groupsize is None and has_zero_points:
        pytest.skip("weight_zero_points is only supported for groupwise quantization")
    device = "cuda"
    rtol, atol = (5e-3, 1e-2 if not has_global_scale else 2e-2)
    # set seed
    torch.random.manual_seed(bits)
    w = torch.randint(-128, 127, (out_features // (8 // bits), in_features), dtype=torch.int8, device=device)
    w_processed = preprocess_weight(w, bits)
    scales_shape = (out_features,) if groupsize is None else (in_features // groupsize, out_features)
    scales = torch.randn(*scales_shape, dtype=torch.float16, device=device) / 128 / math.sqrt(in_features)
    zero_points = torch.randn(*scales_shape, dtype=torch.float16, device=device) / math.sqrt(in_features) if has_zero_points else None
    global_scale = torch.randn(1).item() if has_global_scale else 1.0
    bias = torch.randn(out_features, dtype=torch.float16, device=device) if has_bias else None
    x = torch.randn(batch, in_features, dtype=torch.float16, device=device)
    out = quant_matmul_fn(x, w_processed, scales, zero_points, global_scale=global_scale, bias=bias, bits=bits)
    out_ref = quant_matmul_ref(x, w, scales, zero_points, bias=bias, global_scale=global_scale, bits=bits)
    print(f"Max error: {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol)
