import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from einops import rearrange

from quant_matmul import preprocess_weight, quant_matmul_fn


torch.manual_seed(0)
repeats = 30
dtype = torch.float16
device = 'cuda'

batch = 1
bits = 4
k, n = 4096 * 2, 4096 * 2
# k, n = 4096, 4096
wfp16 = torch.randn(n, k, dtype=dtype, device=device)
w = torch.randint(-128, 127, (k, n // (8 // bits)), dtype=torch.int8, device=device)
wscale = torch.ones(n, dtype=dtype, device=device)
x = torch.randn(batch, k, dtype=dtype, device=device)
# bias = torch.randn(n, dtype=dtype, device=device)
bias = None
wpacked = preprocess_weight(w, bits)
pytorch_profiler(F.linear, x, wfp16)
pytorch_profiler(quant_matmul_fn, x, wpacked, wscale, 1.0, bias, bits)
pytorch_profiler(torch.clone, wpacked)
