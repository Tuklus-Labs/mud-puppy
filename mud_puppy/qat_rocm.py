import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module


class QATLinear(nn.Module):
    """Quantization-aware training linear layer."""

    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = linear.bias

        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_zero_point = nn.Parameter(torch.zeros(1))

    def quantize_weight(self):
        scale = self.weight_scale
        zero_point = torch.round(self.weight_zero_point)
        return torch.fake_quantize_per_tensor_affine(
            self.weight, scale, zero_point, self.qmin, self.qmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.quantize_weight() if self.training else self.weight
        return F.linear(x, weight, self.bias)



def apply_qat(model: nn.Module, bits: int = 8) -> nn.Module:
    """Replace ``nn.Linear`` layers with :class:`QATLinear`."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            qat = QATLinear(module, bits=bits)
            _set_module(model, name, qat)
    return model


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert :class:`QATLinear` layers to int8 ``nn.Linear`` layers."""
    for name, module in list(model.named_modules()):
        if isinstance(module, QATLinear):
            qweight = module.quantize_weight().to(torch.int8)
            linear = nn.Linear(module.in_features, module.out_features)
            linear.weight = nn.Parameter(qweight.float())
            linear.bias = module.bias
            _set_module(model, name, linear)
    return model
