import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from einops import rearrange

from quant_matmul import moe_mlp
from quant_matmul.moe_interface import moe_mlp_ref


torch.manual_seed(0)
repeats = 30
dtype = torch.float16
device = 'cuda'

batch = 1
# k, n = 4096 * 2, 4096 * 2
k, n = 4096, 11008
num_experts = 8
num_active_experts = 2
fc1_weights = torch.randn(num_experts, 2 * n, k, device=device, dtype=dtype) / math.sqrt(k)
fc2_weights = torch.randn(num_experts, k, n, device=device, dtype=dtype) / math.sqrt(n)
gating = torch.randn(batch, num_experts, device=device, dtype=dtype)
x = torch.randn(batch, k, dtype=torch.float16, device=device)

pytorch_profiler(moe_mlp_ref, x, fc1_weights, fc2_weights, gating, num_active_experts)
pytorch_profiler(torch.clone, fc1_weights)
pytorch_profiler(torch.clone, fc2_weights)
pytorch_profiler(moe_mlp, x, fc1_weights, fc2_weights, gating, num_active_experts)
