# Copyright (c) 2024, Tri Dao.

import math
import pytest

import torch

from quant_matmul import moe_mlp
from quant_matmul.moe_interface import moe_mlp_ref


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 16, 37])
# @pytest.mark.parametrize("batch", [1, 2, 3, 4])
# @pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("out_features", [64, 192, 2048, 2752, 4096, 5120])
# @pytest.mark.parametrize("out_features", [64])
@pytest.mark.parametrize("in_features", [192, 2048, 2560, 4096, 5120])
# @pytest.mark.parametrize("in_features", [192])
def test_mlp(in_features, out_features, batch):
    device = "cuda"
    dtype = torch.float16
    rtol, atol = (5e-3, 2e-2)
    # set seed
    torch.random.manual_seed(2357)
    num_experts = 11
    num_active_experts = 3
    fc1_weights = torch.randn(num_experts, 2 * out_features, in_features, device=device, dtype=dtype) / math.sqrt(in_features)
    fc2_weights = torch.randn(num_experts, in_features, out_features, device=device, dtype=dtype) / math.sqrt(out_features)
    gating = torch.randn(batch, num_experts, device=device, dtype=dtype)
    x = torch.randn(batch, in_features, dtype=torch.float16, device=device)
    out = moe_mlp(x, fc1_weights, fc2_weights, gating, num_active_experts)
    out_ref = moe_mlp_ref(x, fc1_weights, fc2_weights, gating, num_active_experts)
    print(f"Max error: {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol)
