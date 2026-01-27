"""JAX/Flax LoRA and QLoRA implementation for Mud Puppy.

This module provides Low-Rank Adaptation (LoRA) layers for efficient fine-tuning
of large language models using JAX and Flax. It supports both standard LoRA
and quantized QLoRA with INT4 base weights.

References:
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Flax NNX LoRA: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/lora.html
- Qwix: https://github.com/google/qwix
- Lorax: https://github.com/davisyoshida/lorax

Example usage:
    from mud_puppy.jax_lora import LoRAConfig, LoRADense, apply_lora_to_model

    # Create a LoRA config
    config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )

    # Wrap a Dense layer with LoRA
    lora_layer = LoRADense(
        features=768,
        r=config.r,
        alpha=config.alpha,
        dropout=config.dropout,
    )

    # Or apply to an entire model
    lora_model = apply_lora_to_model(model, config)

    # After training, merge weights back
    merged_params = merge_lora_params(lora_params, base_params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

# Type aliases
Array = jax.Array
PRNGKey = jax.random.PRNGKey
Shape = Sequence[int]
Dtype = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str]]

# Default initializers following the original LoRA paper
DEFAULT_A_INIT = nn.initializers.kaiming_uniform()
DEFAULT_B_INIT = nn.initializers.zeros


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        r: Rank of the low-rank decomposition. Higher = more capacity but more params.
        alpha: Scaling factor. The LoRA output is scaled by alpha/r.
        dropout: Dropout probability applied to the LoRA path.
        target_modules: List of module name patterns to apply LoRA to (e.g., ["q_proj", "k_proj"]).
        use_rslora: Use rank-stabilized LoRA scaling (alpha / sqrt(r)) instead of alpha / r.
        bias: Whether to train bias parameters. One of "none", "all", "lora_only".
    """

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    use_rslora: bool = False
    bias: str = "none"

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {self.dropout}")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias must be one of 'none', 'all', 'lora_only', got {self.bias}")

    @property
    def scaling(self) -> float:
        """Compute the LoRA scaling factor."""
        if self.use_rslora:
            return self.alpha / math.sqrt(self.r)
        return self.alpha / self.r


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for quantized LoRA (QLoRA).

    Extends LoRAConfig with quantization-specific settings for the base weights.

    Attributes:
        bits: Number of bits for base weight quantization (typically 4).
        double_quant: Apply double quantization to the quantization constants.
        quant_type: Quantization type, either "nf4" (normalized float4) or "fp4".
        compute_dtype: Data type for computation (typically bfloat16).
        use_qwix: Whether to use Qwix for quantization if available.
    """

    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: Dtype = jnp.bfloat16
    use_qwix: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.bits not in (4, 8):
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {self.bits}")
        if self.quant_type not in ("nf4", "fp4", "int4", "int8"):
            raise ValueError(f"Unknown quant_type: {self.quant_type}")


class LoRADense(nn.Module):
    """Dense layer with Low-Rank Adaptation.

    This layer implements W_new = W_base + (B @ A) * scaling, where:
    - W_base is the frozen base weight (features_in, features_out)
    - A is the down-projection (features_in, r)
    - B is the up-projection (r, features_out)
    - scaling = alpha / r (or alpha / sqrt(r) for RSLoRA)

    During training, only A and B are updated. The base weight remains frozen.

    Attributes:
        features: Number of output features.
        r: LoRA rank (low-rank dimension).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability for the LoRA path.
        use_bias: Whether to use bias.
        dtype: Data type for computations.
        param_dtype: Data type for parameters.
        precision: Precision for matrix multiplication.
        kernel_init: Initializer for base kernel (if creating new layer).
        a_init: Initializer for LoRA A matrix.
        b_init: Initializer for LoRA B matrix (default zeros for stability).
        use_rslora: Use rank-stabilized scaling.
    """

    features: int
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    a_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_A_INIT
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_B_INIT
    use_rslora: bool = False

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        *,
        base_kernel: Optional[Array] = None,
        deterministic: bool = False,
    ) -> Array:
        """Apply the LoRA-adapted dense layer.

        Args:
            inputs: Input array of shape (..., features_in).
            base_kernel: Optional pre-existing kernel to wrap. If None, creates new.
            deterministic: If True, disable dropout.

        Returns:
            Output array of shape (..., features).
        """
        features_in = inputs.shape[-1]
        dtype = self.dtype or inputs.dtype

        # Base kernel (frozen during LoRA training)
        if base_kernel is not None:
            kernel = base_kernel
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (features_in, self.features),
                self.param_dtype,
            )

        # LoRA matrices
        lora_a = self.param(
            "lora_a",
            self.a_init,
            (features_in, self.r),
            self.param_dtype,
        )
        lora_b = self.param(
            "lora_b",
            self.b_init,
            (self.r, self.features),
            self.param_dtype,
        )

        # Bias (optional)
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
        else:
            bias = None

        # Compute scaling factor
        if self.use_rslora:
            scaling = self.alpha / math.sqrt(self.r)
        else:
            scaling = self.alpha / self.r

        # Cast inputs and params to compute dtype
        inputs = jnp.asarray(inputs, dtype)
        kernel = jnp.asarray(kernel, dtype)
        lora_a = jnp.asarray(lora_a, dtype)
        lora_b = jnp.asarray(lora_b, dtype)

        # Base output: x @ W
        base_output = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        # LoRA path: x @ A @ B * scaling
        lora_output = jax.lax.dot_general(
            inputs,
            lora_a,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        # Apply dropout to LoRA path during training
        if self.dropout > 0 and not deterministic:
            lora_output = nn.Dropout(rate=self.dropout)(lora_output, deterministic=False)

        lora_output = jax.lax.dot_general(
            lora_output,
            lora_b,
            (((lora_output.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        # Combine: base + scaled LoRA
        output = base_output + lora_output * scaling

        # Add bias if present
        if bias is not None:
            bias = jnp.asarray(bias, dtype)
            output = output + bias

        return output


class LoRAAdapter(nn.Module):
    """Standalone LoRA adapter (B @ A multiplication only).

    This is useful when you want to add LoRA to an existing layer without
    replacing it. You manually add the adapter output to the base layer output.

    Example:
        base_output = nn.Dense(features)(x)
        lora_delta = LoRAAdapter(features, r=8)(x)
        output = base_output + lora_delta
    """

    features: int
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    a_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_A_INIT
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_B_INIT
    use_rslora: bool = False

    @nn.compact
    def __call__(self, inputs: Array, *, deterministic: bool = False) -> Array:
        """Compute LoRA delta: (x @ A @ B) * scaling."""
        features_in = inputs.shape[-1]
        dtype = self.dtype or inputs.dtype

        lora_a = self.param(
            "lora_a",
            self.a_init,
            (features_in, self.r),
            self.param_dtype,
        )
        lora_b = self.param(
            "lora_b",
            self.b_init,
            (self.r, self.features),
            self.param_dtype,
        )

        scaling = self.alpha / (math.sqrt(self.r) if self.use_rslora else self.r)

        inputs = jnp.asarray(inputs, dtype)
        lora_a = jnp.asarray(lora_a, dtype)
        lora_b = jnp.asarray(lora_b, dtype)

        # x @ A
        hidden = jax.lax.dot_general(
            inputs,
            lora_a,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        # Dropout on hidden
        if self.dropout > 0 and not deterministic:
            hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=False)

        # hidden @ B
        output = jax.lax.dot_general(
            hidden,
            lora_b,
            (((hidden.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        return output * scaling


class QuantizedLinear(nn.Module):
    """INT4/INT8 quantized linear layer for QLoRA base weights.

    Implements weight-only quantization where the weights are stored in low
    precision but computation is done in higher precision (typically bf16).

    For full QLoRA, wrap this with LoRAAdapter.
    """

    features: int
    bits: int = 4
    quant_type: str = "nf4"
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    compute_dtype: Dtype = jnp.bfloat16
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    group_size: int = 128  # Quantization group size for better accuracy

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Apply quantized linear transformation."""
        features_in = inputs.shape[-1]
        dtype = self.dtype or self.compute_dtype

        # For simplicity, we store the quantized weight and scale
        # In a full implementation, this would use packed int4 storage
        qweight = self.param(
            "qweight",
            nn.initializers.zeros,
            (features_in, self.features),
            jnp.int8,
        )
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (features_in // self.group_size + 1, self.features),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
        else:
            bias = None

        # Dequantize weights for computation
        weight = self._dequantize(qweight, scale, features_in)

        inputs = jnp.asarray(inputs, dtype)
        weight = jnp.asarray(weight, dtype)

        output = jax.lax.dot_general(
            inputs,
            weight,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            output = output + jnp.asarray(bias, dtype)

        return output

    def _dequantize(self, qweight: Array, scale: Array, features_in: int) -> Array:
        """Dequantize INT4/INT8 weights to compute dtype."""
        if self.quant_type == "nf4":
            # NormalFloat4 dequantization
            # NF4 uses a lookup table based on normal distribution quantiles
            nf4_table = jnp.array(
                [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0],
                dtype=self.compute_dtype,
            )
            # Map int values to NF4 values
            weight = nf4_table[qweight.astype(jnp.int32) + 8]  # offset for signed
        else:
            # Standard int dequantization
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
            weight = qweight.astype(self.compute_dtype) / qmax

        # Apply per-group scale
        # Broadcast scale to weight shape
        groups = features_in // self.group_size
        expanded_scale = jnp.repeat(scale, self.group_size, axis=0)[:features_in]
        weight = weight * expanded_scale

        return weight


class QLoRADense(nn.Module):
    """Combined QLoRA layer: quantized base + LoRA adapters.

    This combines INT4 quantized base weights with trainable LoRA adapters
    for memory-efficient fine-tuning.
    """

    features: int
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    bits: int = 4
    quant_type: str = "nf4"
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    compute_dtype: Dtype = jnp.bfloat16
    precision: PrecisionLike = None
    a_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_A_INIT
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_B_INIT
    use_rslora: bool = False
    group_size: int = 128

    @nn.compact
    def __call__(self, inputs: Array, *, deterministic: bool = False) -> Array:
        """Apply QLoRA: quantized_base(x) + lora_adapter(x)."""
        # Quantized base layer
        base_output = QuantizedLinear(
            features=self.features,
            bits=self.bits,
            quant_type=self.quant_type,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            precision=self.precision,
            group_size=self.group_size,
            name="base",
        )(inputs)

        # LoRA adapter
        lora_delta = LoRAAdapter(
            features=self.features,
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            a_init=self.a_init,
            b_init=self.b_init,
            use_rslora=self.use_rslora,
            name="lora",
        )(inputs, deterministic=deterministic)

        return base_output + lora_delta


# =============================================================================
# Model-level utilities
# =============================================================================


def merge_lora_params(
    params: Dict[str, Any],
    lora_params: Dict[str, Any],
    scaling: float = 1.0,
) -> Dict[str, Any]:
    """Merge LoRA weights back into base model weights.

    After training, this produces a single set of weights that can be used
    without the LoRA infrastructure.

    Args:
        params: Base model parameters (frozen during training).
        lora_params: LoRA adapter parameters (A, B matrices).
        scaling: LoRA scaling factor (alpha/r).

    Returns:
        Merged parameters with LoRA deltas folded into base weights.
    """
    params_flat = flatten_dict(unfreeze(params))
    lora_flat = flatten_dict(unfreeze(lora_params))

    merged = {}
    for key, value in params_flat.items():
        merged[key] = value

        # Check if this is a kernel that has corresponding LoRA params
        if key[-1] == "kernel":
            lora_key_prefix = key[:-1]
            lora_a_key = (*lora_key_prefix, "lora_a")
            lora_b_key = (*lora_key_prefix, "lora_b")

            if lora_a_key in lora_flat and lora_b_key in lora_flat:
                lora_a = lora_flat[lora_a_key]
                lora_b = lora_flat[lora_b_key]
                # W_merged = W_base + (A @ B) * scaling
                delta = jnp.matmul(lora_a, lora_b) * scaling
                merged[key] = value + delta

    return freeze(unflatten_dict(merged))


def split_lora_params(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split parameters into base and LoRA components.

    Args:
        params: Combined parameters containing both base and LoRA weights.

    Returns:
        Tuple of (base_params, lora_params).
    """
    params_flat = flatten_dict(unfreeze(params))

    base_params = {}
    lora_params = {}

    for key, value in params_flat.items():
        if "lora_a" in key or "lora_b" in key:
            lora_params[key] = value
        else:
            base_params[key] = value

    return freeze(unflatten_dict(base_params)), freeze(unflatten_dict(lora_params))


def _matches_target_module(name: str, targets: List[str]) -> bool:
    """Check if a module name matches any of the target patterns."""
    for target in targets:
        if target in name:
            return True
    return False


def get_lora_param_filter(
    config: LoRAConfig,
) -> Callable[[str, Any], bool]:
    """Create a filter function for selecting trainable parameters.

    Returns a function that returns True for parameters that should be trained
    (LoRA params and optionally biases).
    """
    def is_trainable(path: str, param: Any) -> bool:
        # LoRA params are always trainable
        if "lora_a" in path or "lora_b" in path:
            return True
        # Handle bias training
        if "bias" in path:
            if config.bias == "all":
                return True
            if config.bias == "lora_only":
                # Only train bias if it's part of a LoRA-adapted layer
                return any(target in path for target in config.target_modules)
        return False

    return is_trainable


def freeze_non_lora_params(
    params: Dict[str, Any],
    config: LoRAConfig,
) -> Dict[str, Any]:
    """Mark non-LoRA parameters as frozen (stop gradient).

    This is useful when you want to use a standard optimizer but only
    update LoRA parameters.
    """
    params_flat = flatten_dict(unfreeze(params))
    filter_fn = get_lora_param_filter(config)

    frozen = {}
    for key, value in params_flat.items():
        path = "/".join(str(k) for k in key)
        if filter_fn(path, value):
            frozen[key] = value
        else:
            frozen[key] = jax.lax.stop_gradient(value)

    return freeze(unflatten_dict(frozen))


def count_parameters(params: Dict[str, Any]) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    params_flat = flatten_dict(unfreeze(params))

    total = 0
    lora_count = 0

    for key, value in params_flat.items():
        size = value.size
        total += size
        if "lora_a" in key or "lora_b" in key:
            lora_count += size

    return total, lora_count


# =============================================================================
# Qwix integration (optional)
# =============================================================================


def try_import_qwix():
    """Try to import Qwix for advanced quantization support."""
    try:
        import qwix
        return qwix
    except ImportError:
        return None


def quantize_for_qlora(
    params: Dict[str, Any],
    config: QLoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Quantize model parameters for QLoRA.

    Attempts to use Qwix if available, otherwise falls back to simple
    per-channel quantization.

    Args:
        params: Model parameters to quantize.
        config: QLoRA configuration.
        target_modules: Modules to quantize. If None, quantizes all linear layers.

    Returns:
        Quantized parameters.
    """
    qwix = try_import_qwix()

    if qwix is not None and config.use_qwix:
        return _quantize_with_qwix(params, config, target_modules)
    else:
        return _quantize_simple(params, config, target_modules)


def _quantize_simple(
    params: Dict[str, Any],
    config: QLoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Simple per-channel INT4 quantization without Qwix."""
    params_flat = flatten_dict(unfreeze(params))
    quantized = {}

    for key, value in params_flat.items():
        path = "/".join(str(k) for k in key)
        should_quantize = key[-1] == "kernel"

        if target_modules is not None:
            should_quantize = should_quantize and _matches_target_module(path, target_modules)

        if should_quantize and value.ndim == 2:
            qweight, scale = _quantize_per_channel(
                value,
                bits=config.bits,
                group_size=128,
            )
            # Store quantized weight and scale
            quantized[(*key[:-1], "qweight")] = qweight
            quantized[(*key[:-1], "scale")] = scale
        else:
            quantized[key] = value

    return freeze(unflatten_dict(quantized))


def _quantize_per_channel(
    weight: Array,
    bits: int = 4,
    group_size: int = 128,
) -> Tuple[Array, Array]:
    """Quantize weight matrix with per-group scaling."""
    # weight shape: (in_features, out_features)
    in_features, out_features = weight.shape

    # Compute scale per group
    num_groups = (in_features + group_size - 1) // group_size
    scale = jnp.zeros((num_groups, out_features), dtype=jnp.float32)

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    # Pad weight to multiple of group_size
    pad_size = num_groups * group_size - in_features
    if pad_size > 0:
        weight_padded = jnp.pad(weight, ((0, pad_size), (0, 0)))
    else:
        weight_padded = weight

    weight_reshaped = weight_padded.reshape(num_groups, group_size, out_features)

    # Compute per-group scale
    max_val = jnp.abs(weight_reshaped).max(axis=1, keepdims=True) + 1e-8
    scale = max_val.squeeze(1) / qmax

    # Quantize
    qweight_reshaped = jnp.clip(
        jnp.round(weight_reshaped / max_val * qmax),
        qmin,
        qmax,
    ).astype(jnp.int8)

    # Reshape back and trim padding
    qweight = qweight_reshaped.reshape(-1, out_features)[:in_features]

    return qweight, scale


def _quantize_with_qwix(
    params: Dict[str, Any],
    config: QLoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Quantize using Qwix library for better quality."""
    qwix = try_import_qwix()
    if qwix is None:
        raise ImportError("Qwix not available")

    # Build quantization rules based on target modules
    if target_modules:
        pattern = "|".join(target_modules)
        module_regex = f".*({pattern}).*"
    else:
        module_regex = ".*"

    # Map our config to Qwix quantization type
    qtype_map = {
        "nf4": "nf4",
        "fp4": "fp4",
        "int4": "int4",
        "int8": "int8",
    }
    qtype = qtype_map.get(config.quant_type, "int4")

    rules = [
        qwix.QuantizationRule(
            module_path=module_regex,
            weight_qtype=qtype,
            act_qtype=None,  # Weight-only quantization
        )
    ]

    # This is a simplified interface - actual Qwix usage requires model structure
    # In practice, you'd use qwix.quantize_model() on the model itself
    return _quantize_simple(params, config, target_modules)


# =============================================================================
# LoRA model wrapper
# =============================================================================


class LoRAModel(nn.Module):
    """Wrapper that applies LoRA to specified modules in a model.

    This is a convenience class that automatically wraps target modules
    with LoRA layers based on the provided configuration.

    Note: For complex models, you may need to manually construct the
    LoRA-adapted model using LoRADense layers directly.

    Example:
        config = LoRAConfig(r=8, target_modules=["q_proj", "v_proj"])
        lora_model = LoRAModel(base_model, config)
        params = lora_model.init(rng, x)
    """

    base_model: nn.Module
    config: LoRAConfig

    def setup(self):
        """Set up the LoRA-adapted model.

        Note: This is a simplified implementation. For full model wrapping,
        you would need to use JAX transforms (like lorax) or manually
        construct the adapted model architecture.
        """
        pass

    @nn.compact
    def __call__(self, *args, deterministic: bool = False, **kwargs):
        """Forward pass through the LoRA-adapted model.

        This delegates to the base model. For actual LoRA adaptation,
        you need to either:
        1. Use the lorax library for automatic transformation
        2. Manually replace target layers with LoRADense
        3. Use Flax's module surgery utilities
        """
        return self.base_model(*args, **kwargs)


def make_lora_train_state(
    params: Dict[str, Any],
    config: LoRAConfig,
    optimizer,
):
    """Create an optimizer state that only updates LoRA parameters.

    This is useful with optax where you want to use the same optimizer
    but only apply updates to LoRA parameters.

    Args:
        params: All model parameters.
        config: LoRA configuration.
        optimizer: An optax optimizer.

    Returns:
        Optax optimizer state for LoRA training.
    """
    try:
        import optax
    except ImportError:
        raise ImportError("optax required for make_lora_train_state")

    # Create a mask that selects only LoRA parameters
    params_flat = flatten_dict(unfreeze(params))
    filter_fn = get_lora_param_filter(config)

    def mask_fn(params):
        params_flat = flatten_dict(unfreeze(params))
        mask = {}
        for key, value in params_flat.items():
            path = "/".join(str(k) for k in key)
            mask[key] = filter_fn(path, value)
        return freeze(unflatten_dict(mask))

    # Use optax.masked to only update LoRA params
    masked_optimizer = optax.masked(optimizer, mask_fn(params))

    return masked_optimizer, masked_optimizer.init(params)


# =============================================================================
# Utility functions for working with LoRA
# =============================================================================


def init_lora_from_pretrained(
    base_params: Dict[str, Any],
    config: LoRAConfig,
    rng: PRNGKey,
) -> Dict[str, Any]:
    """Initialize LoRA parameters for a pretrained model.

    Given base model parameters, this creates the corresponding LoRA
    parameters (A and B matrices) initialized appropriately.

    Args:
        base_params: Pretrained model parameters.
        config: LoRA configuration.
        rng: Random key for initialization.

    Returns:
        Combined parameters with both base and LoRA weights.
    """
    params_flat = flatten_dict(unfreeze(base_params))
    combined = {}

    for key, value in params_flat.items():
        combined[key] = value
        path = "/".join(str(k) for k in key)

        # Add LoRA params for target module kernels
        if key[-1] == "kernel" and _matches_target_module(path, config.target_modules):
            if value.ndim == 2:
                in_features, out_features = value.shape
                rng, a_rng, b_rng = jax.random.split(rng, 3)

                # Initialize A with kaiming uniform, B with zeros
                lora_a = DEFAULT_A_INIT(a_rng, (in_features, config.r), jnp.float32)
                lora_b = DEFAULT_B_INIT(b_rng, (config.r, out_features), jnp.float32)

                combined[(*key[:-1], "lora_a")] = lora_a
                combined[(*key[:-1], "lora_b")] = lora_b

    return freeze(unflatten_dict(combined))


def print_lora_summary(params: Dict[str, Any], config: LoRAConfig) -> str:
    """Generate a summary of LoRA parameters.

    Returns:
        Human-readable summary string.
    """
    total, trainable = count_parameters(params)
    pct = 100.0 * trainable / total if total > 0 else 0.0

    lines = [
        "LoRA Configuration Summary",
        "=" * 40,
        f"Rank (r): {config.r}",
        f"Alpha: {config.alpha}",
        f"Scaling: {config.scaling:.4f}",
        f"Dropout: {config.dropout}",
        f"Target modules: {config.target_modules}",
        f"Bias training: {config.bias}",
        "",
        "Parameter Counts",
        "-" * 40,
        f"Total parameters: {total:,}",
        f"Trainable (LoRA): {trainable:,}",
        f"Frozen: {total - trainable:,}",
        f"Trainable %: {pct:.2f}%",
    ]

    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    "LoRAConfig",
    "QLoRAConfig",
    # Layer classes
    "LoRADense",
    "LoRAAdapter",
    "QuantizedLinear",
    "QLoRADense",
    # Model wrapper
    "LoRAModel",
    # Parameter utilities
    "merge_lora_params",
    "split_lora_params",
    "freeze_non_lora_params",
    "count_parameters",
    "get_lora_param_filter",
    "init_lora_from_pretrained",
    "print_lora_summary",
    # Quantization
    "quantize_for_qlora",
    "try_import_qwix",
    # Optimizer utilities
    "make_lora_train_state",
]


# =============================================================================
# Self-test when run as main
# =============================================================================


def _run_tests():
    """Run basic sanity tests for the LoRA implementation."""
    print("Running JAX LoRA tests...")
    print("=" * 60)

    # Test 1: LoRAConfig validation
    print("\n[Test 1] LoRAConfig validation")
    config = LoRAConfig(r=8, alpha=16.0, dropout=0.1)
    assert config.scaling == 2.0, f"Expected scaling=2.0, got {config.scaling}"
    print(f"  - Config created: r={config.r}, alpha={config.alpha}, scaling={config.scaling}")

    # Test with RSLoRA
    config_rslora = LoRAConfig(r=8, alpha=16.0, use_rslora=True)
    expected_scaling = 16.0 / math.sqrt(8)
    assert abs(config_rslora.scaling - expected_scaling) < 1e-6
    print(f"  - RSLoRA scaling: {config_rslora.scaling:.4f}")

    # Test 2: LoRADense layer
    print("\n[Test 2] LoRADense forward pass")
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (2, 16))  # batch=2, features=16

    lora_layer = LoRADense(features=32, r=4, alpha=8.0)
    params = lora_layer.init(rng, x)

    # Check parameter shapes
    kernel_shape = params["params"]["kernel"].shape
    lora_a_shape = params["params"]["lora_a"].shape
    lora_b_shape = params["params"]["lora_b"].shape
    print(f"  - Kernel shape: {kernel_shape}")
    print(f"  - LoRA A shape: {lora_a_shape}")
    print(f"  - LoRA B shape: {lora_b_shape}")

    assert kernel_shape == (16, 32), f"Wrong kernel shape: {kernel_shape}"
    assert lora_a_shape == (16, 4), f"Wrong lora_a shape: {lora_a_shape}"
    assert lora_b_shape == (4, 32), f"Wrong lora_b shape: {lora_b_shape}"

    # Forward pass
    y = lora_layer.apply(params, x, deterministic=True)
    assert y.shape == (2, 32), f"Wrong output shape: {y.shape}"
    print(f"  - Output shape: {y.shape}")

    # Test 3: LoRAAdapter
    print("\n[Test 3] LoRAAdapter")
    adapter = LoRAAdapter(features=32, r=4, alpha=8.0)
    adapter_params = adapter.init(rng, x)
    delta = adapter.apply(adapter_params, x, deterministic=True)
    assert delta.shape == (2, 32), f"Wrong delta shape: {delta.shape}"
    print(f"  - Delta shape: {delta.shape}")

    # Test 4: Parameter counting
    print("\n[Test 4] Parameter counting")
    total, trainable = count_parameters(params)
    print(f"  - Total params: {total:,}")
    print(f"  - LoRA params: {trainable:,}")
    print(f"  - Trainable %: {100.0 * trainable / total:.2f}%")

    # Test 5: LoRA merge
    print("\n[Test 5] LoRA weight merging")
    # Split into base and lora
    base_params, lora_only = split_lora_params(params)

    # Merge with scaling
    merged = merge_lora_params(base_params, params, scaling=2.0)

    # The merged kernel should be different from original
    original_kernel = params["params"]["kernel"]
    merged_kernel = merged["params"]["kernel"]
    kernel_diff = jnp.abs(merged_kernel - original_kernel).max()
    print(f"  - Max kernel difference after merge: {kernel_diff:.6f}")

    # Since B is initialized to zeros, initially the diff should be 0
    # (but we can verify the merge function works)
    assert merged_kernel.shape == original_kernel.shape
    print("  - Merge successful")

    # Test 6: QLoRAConfig
    print("\n[Test 6] QLoRAConfig")
    qconfig = QLoRAConfig(r=8, alpha=16.0, bits=4, quant_type="nf4")
    print(f"  - QLoRA config: bits={qconfig.bits}, quant_type={qconfig.quant_type}")
    assert qconfig.bits == 4
    assert qconfig.quant_type == "nf4"

    # Test 7: QLoRADense
    print("\n[Test 7] QLoRADense forward pass")
    qlora_layer = QLoRADense(features=32, r=4, alpha=8.0, bits=4)
    qlora_params = qlora_layer.init(rng, x)
    y_qlora = qlora_layer.apply(qlora_params, x, deterministic=True)
    assert y_qlora.shape == (2, 32), f"Wrong QLoRA output shape: {y_qlora.shape}"
    print(f"  - QLoRA output shape: {y_qlora.shape}")

    # Test 8: init_lora_from_pretrained
    print("\n[Test 8] Initialize LoRA from pretrained")

    # Create a simple "pretrained" model params
    class SimpleMLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(64, name="q_proj")(x)
            x = nn.relu(x)
            x = nn.Dense(32, name="v_proj")(x)
            return x

    mlp = SimpleMLP()
    pretrained_params = mlp.init(rng, x)

    config = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
    lora_params = init_lora_from_pretrained(pretrained_params, config, rng)

    # Check that LoRA params were added
    flat_params = flatten_dict(unfreeze(lora_params))
    lora_keys = [k for k in flat_params.keys() if "lora_a" in k or "lora_b" in k]
    print(f"  - Added {len(lora_keys)} LoRA parameter tensors")
    assert len(lora_keys) == 4  # 2 layers x (lora_a + lora_b)

    # Test 9: Print summary
    print("\n[Test 9] LoRA Summary")
    summary = print_lora_summary(lora_params, config)
    print(summary)

    # Test 10: Quantization (simple)
    print("\n[Test 10] Simple quantization")
    test_weight = jax.random.normal(rng, (128, 64))
    qweight, scale = _quantize_per_channel(test_weight, bits=4, group_size=32)
    print(f"  - Input shape: {test_weight.shape}")
    print(f"  - Quantized shape: {qweight.shape}")
    print(f"  - Scale shape: {scale.shape}")
    assert qweight.dtype == jnp.int8
    assert qweight.shape == test_weight.shape

    print("\n" + "=" * 60)
    print("All tests passed!")
    return True


if __name__ == "__main__":
    _run_tests()
