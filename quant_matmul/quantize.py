import torch
import torch.nn as nn

from quant_matmul import preprocess_weight, quant_matmul_fn


def quantize_symmetric(weight, bits=4):
    """
    Arguments:
        weight: (out_features, in_features), fp16 or bf16 or fp32
    Return:
        qweight: (out_features, in_features) if 8 bits, (out_features / 2, in_features) if int4.
            Stored in torch.int8.
        scales: (out_features,), fp16
    """
    assert weight.ndim == 2
    assert bits in [4, 8]
    max_val = weight.abs().amax(dim=1)
    scales = max_val / (127.0 if bits == 8 else 7.0)
    scales = scales.clamp(min=torch.finfo(torch.float32).eps).to(torch.float16)
    qweight = (weight / scales[:, None]).round().to(torch.int8)
    if bits == 4:
        qweight = qweight[::2] | (qweight[1::2] << 4)
    return qweight, scales


class WeightOnlyInt8Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        assert in_features % 64 == 0 and out_features % 4 == 0
        self.register_buffer("weight", torch.empty(out_features, in_features, device=device, dtype=torch.int8))
        self.register_buffer("scales", torch.empty(out_features, device=device, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, device=device, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear):
        weight = linear.weight
        wlinear = cls(weight.shape[1], weight.shape[0], linear.bias is not None, weight.device)
        with torch.no_grad():
            qweight, scales = quantize_symmetric(weight, bits=8)
            wlinear.weight.copy_(preprocess_weight(qweight, bits=8))
            wlinear.scales.copy_(scales)
            if linear.bias is not None:
                wlinear.bias.copy_(linear.bias.detach())
        return wlinear

    def forward(self, x):
        return quant_matmul_fn(x, self.weight, scales=self.scales, bias=self.bias, bits=8)


def replace_linear_weight_only_int8(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt8Linear.from_linear(child))
        else:
            replace_linear_weight_only_int8(child)


class WeightOnlyInt4Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        assert in_features % 64 == 0 and out_features % 2 == 0
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, device=device, dtype=torch.int8))
        self.register_buffer("scales", torch.empty(out_features, device=device, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, device=device, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear):
        weight = linear.weight
        wlinear = cls(weight.shape[1], weight.shape[0], linear.bias is not None, weight.device)
        with torch.no_grad():
            qweight, scales = quantize_symmetric(weight, bits=4)
            wlinear.weight.copy_(preprocess_weight(qweight, bits=4))
            wlinear.scales.copy_(scales)
            if linear.bias is not None:
                wlinear.bias.copy_(linear.bias.detach())
        return wlinear

    def forward(self, x):
        return quant_matmul_fn(x, self.weight, scales=self.scales, bias=self.bias, bits=4)


def replace_linear_weight_only_int4(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt4Linear.from_linear(child))
        else:
            replace_linear_weight_only_int4(child)


