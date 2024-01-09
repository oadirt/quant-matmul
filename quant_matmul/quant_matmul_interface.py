import torch
import torch.nn.functional as F

from einops import rearrange, repeat

import quant_matmul_cuda


def preprocess_weight(weight, bits=4):
    """
    Arguments:
        weight: (n, k) if bits == 8, (n / 2, k) if bits == 4, in int8
    Returns:
        packed_weight: (n, k) if bits == 8, (n, k / 2) if bits == 4, in int8
    """
    assert bits in [4, 8]
    assert weight.dtype == torch.int8
    if bits == 4:
        n, k = weight.shape[0] * 2, weight.shape[1]
        assert n % 4 == 0 and k % 64 == 0
        perm = torch.tensor([0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31]).reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(-1).tolist()
        weight = weight.reshape(n // 2, k // 32, 32)[:, :, perm].reshape(n // 2, k)
        weight_8 = torch.stack([(weight << 4) >> 4, weight >> 4], dim=1).reshape(n, k) + 8
        weight = weight_8[:, ::2] | (weight_8[:, 1::2] << 4)
        weight = rearrange(weight, "(nn four) (kk kblock) -> nn (kk four kblock)", four=4, kblock=32).reshape(n, k // 2)
    else:
        n, k = weight.shape
        assert n % 2 == 0 and k % 64 == 0
        perm = torch.tensor([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]).reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1).tolist()
        weight = weight.reshape(n, k // 16, 16)[:, :, perm].reshape(n, k)
        weight = rearrange(weight + 128, "(nn two) (kk kblock) -> nn (kk two kblock)", two=2, kblock=64).reshape(n, k)
    return weight.contiguous()


def quant_matmul_ref(x, quantized_weight, scales=None, zero_points=None, global_scale=1.0, global_bias=0.0,
                     bias=None, bits=4, upcast=False):
    """
    Weight will be dequantized as ((quantized_weight * global_scale + global_bias) * scales + zero_points
    Arguments:
        x: (..., in_features), fp16
        quantized_weight: (out_features, in_features) if bits == 8, (out_features // 2, in_features)
            if bits == 4, stored in int8
        scales: (in_features / group_size, out_features) or (out_features,), fp16
        zero_points: (in_features / group_size, out_features) of (out_features,), fp16.
           (i.e. groupwise quantization).
        global_scale: float
        global_bias: float
        bias: (out_features,), fp16
        bits: 4 or 8
    Return:
        out: (..., out_features), fp16
    """
    assert bits in [4, 8]
    if bits == 4:
        quantized_weight = rearrange(
            torch.stack([(quantized_weight << 4) >> 4, quantized_weight >> 4], dim=1), "o two i -> (o two) i"
        )
    w = quantized_weight.to(torch.float32 if upcast else torch.float16)
    bias = bias.to(torch.float32 if upcast else torch.float16) if bias is not None else None
    x = x.to(torch.float32 if upcast else torch.float16)
    if global_scale != 1.0:
        w *= global_scale
    if global_bias != 0.0:
        w += global_bias
    if scales is not None:
        assert scales.ndim in [1, 2]
        if scales.ndim == 1:
            scales = rearrange(scales, "n -> n 1")
        else:
            groupsize = w.shape[1] // scales.shape[0]
            assert w.shape[1] % groupsize == 0
            scales = repeat(scales, "ngroups n -> n (ngroups groupsize)", groupsize=groupsize)
        w *= scales
    if zero_points is not None:
        if zero_points.ndim == 1:
            zero_points = rearrange(zero_points, "n -> n 1")
        else:
            groupsize = w.shape[1] // zero_points.shape[0]
            assert w.shape[1] % groupsize == 0
            zero_points = repeat(zero_points, "ngroups n -> n (ngroups groupsize)", groupsize=groupsize)
        w += zero_points
    return F.linear(x, w, bias)


def quant_matmul_fn(x, processed_weight, scales=None, zero_points=None, global_scale=1.0, global_bias=0.0,
                    bias=None, bits=4):
    """
    Weight will be dequantized as quantized_weight * global_scale + global_bias
    or quantized_weight * scales + zero_points.
    Arguments:
        x: (..., in_features), fp16
        processed_weight: (out_features, in_features) if bits == 8, (out_features // 2, in_features)
            if bits == 4, stored in int8. The weight is in column-major format, returned by the
            preprocess_weight function.
        scales: (in_features / group_size, out_features) or (out_features,), fp16
        zero_points: (in_features / group_size, out_features), fp16. Only supported if scales is 2D
           (i.e. groupwise quantization).
        global_scale: float
        global_bias: float
        bias: (out_features,), fp16
        bits: 4 or 8
    Return:
        out: (..., out_features), fp16
    """
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    out = quant_matmul_cuda.quant_matmul(x, processed_weight, scales, zero_points, global_scale, global_bias, bias, bits)
    return out.reshape(*x_shape[:-1], out.shape[-1])
