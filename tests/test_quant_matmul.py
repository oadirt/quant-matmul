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
    w = torch.randint(-128, 127, (in_features, out_features // (8 // bits)), dtype=torch.int8, device=device)
    w_processed = preprocess_weight(w, bits)
    wpacked_ref = quant_matmul_cuda.preprocess_weight(w.cpu(), bits, 80).to(device)
    assert torch.equal(w_processed, wpacked_ref)


@pytest.mark.parametrize("bits", [8, 4])
# @pytest.mark.parametrize("bits", [8])
# @pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 16, 37])
# @pytest.mark.parametrize("batch", [1, 2, 3, 4])
# @pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("groupsize", [None, 64, 128])
# @pytest.mark.parametrize("groupsize", [None])
@pytest.mark.parametrize("out_features", [64, 192, 2048, 2752, 4096, 5120])
# @pytest.mark.parametrize("out_features", [256])
@pytest.mark.parametrize("in_features", [192, 2048, 2560, 4096, 5120])
# @pytest.mark.parametrize("in_features", [256])
def test_multiply(in_features, out_features, groupsize, batch, has_bias, bits):
    if groupsize is not None and in_features % groupsize != 0:
        pytest.skip("groupsize must be divisible by in_features")
    device = "cuda"
    atol, rtol = (5e-3, 8e-3)
    # set seed
    torch.random.manual_seed(bits)
    w = torch.randint(-128, 127, (in_features, out_features // (8 // bits)), dtype=torch.int8, device=device)
    w_processed = preprocess_weight(w, bits)
    scales_shape = (out_features,) if groupsize is None else (in_features // groupsize, out_features)
    scales = torch.randn(*scales_shape, dtype=torch.float16, device=device) / 128 / math.sqrt(in_features)
    bias = torch.randn(out_features, dtype=torch.float16, device=device) if has_bias else None
    x = torch.randn(batch, in_features, dtype=torch.float16, device=device)
    out = quant_matmul_fn(x, w_processed, scales, bias=bias, bits=bits)
    out_ref = quant_matmul_ref(x, w.T, scales, bias=bias, bits=bits)
    print(f"Max error: {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol)