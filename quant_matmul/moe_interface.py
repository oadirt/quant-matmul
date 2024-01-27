# Copyright (c) 2024, Tri Dao.

import torch
import torch.nn.functional as F

from einops import rearrange

import quant_matmul_cuda


def moe_mlp(x, fc1_weights, fc2_weights, gating, num_active_experts):
    """
    Arguments:
        x: [batch, in_features]
        fc1_weights: [num_experts, 2 * out_features, in_features]
        fc2_weights: [num_experts, in_features, out_features]
        gating: [batch, num_experts]
        num_active_experts: int, number of experts to use per input
    Return:
        out: [batch, in_features]
    """
    batch = x.shape[0]
    num_experts = fc1_weights.shape[0]
    expert_scales, indices = torch.topk(gating, num_active_experts, dim=-1)
    expert_scales = torch.softmax(expert_scales, dim=-1)
    indices_sorted, perm = torch.sort(indices.flatten(), dim=-1, stable=True)
    x_dup = x[perm // num_active_experts]
    total_rows_before_expert = torch.searchsorted(
        indices_sorted,
        torch.arange(num_experts, device=x.device, dtype=torch.long),
        side="right"
    )
    x1, x2 = quant_matmul_cuda.moe_matmul(x_dup, fc1_weights, total_rows_before_expert).chunk(2, dim=-1)
    y = F.silu(x1) * x2
    y = quant_matmul_cuda.moe_matmul(y, fc2_weights, total_rows_before_expert)
    y_unpermuted = y[torch.argsort(perm)]
    out = torch.einsum("bei,be->bi", rearrange(y_unpermuted, "(b e) i -> b e i", e=num_active_experts), expert_scales)
    # expert_scales_permuted = expert_scales.flatten()[perm]
    # out_permuted = y * rearrange(expert_scales_permuted, "b -> b 1")
    # out = out_permuted.scatter_add(1, torch.arange(batch * num_active_experts, device=device, dtype=torch.int32).unsqueeze(-1), torch.zeros_like(out_permuted))
    return out


def moe_mlp_ref(x, fc1_weights, fc2_weights, gating, num_active_experts):
    """
    Arguments:
        x: [batch, in_features]
        fc1_weights: [num_experts, 2 * out_features, in_features]
        fc2_weights: [num_experts, in_features, out_features]
        gating: [batch, num_experts]
        num_active_experts: int, number of experts to use per input
    Return:
        out: [batch, in_features]
    """
    expert_scales, indices = torch.topk(gating, num_active_experts, dim=-1)
    expert_scales = torch.softmax(expert_scales, dim=-1)
    x1, x2 = torch.einsum("bi,beoi->beo", x, fc1_weights[indices]).chunk(2, dim=-1)
    y = F.silu(x1) * x2
    y = torch.einsum("beo,beio->bei", y, fc2_weights[indices])
    out = torch.einsum("bei,be->bi", y, expert_scales)
    return out
