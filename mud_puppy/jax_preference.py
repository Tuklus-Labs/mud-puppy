"""JAX/Flax implementation of preference-based fine-tuning algorithms.

This module implements DPO, IPO, KTO, and ORPO in pure JAX/Flax, following
the functional programming style of JAX. All loss functions are implemented
as pure functions that can be JIT-compiled and used with any Flax model.

Loss Functions:
    - DPO (Direct Preference Optimization): Uses sigmoid loss on preference margins
    - IPO (Identity Preference Optimization): Uses squared hinge-style loss
    - KTO (Kahneman-Tversky Optimization): Asymmetric loss for single responses
    - ORPO (Odds Ratio Preference Optimization): Combines SFT with odds ratio term

Dataset format (JSON/JSONL):
    - ``prompt``: Input text
    - ``chosen``: Preferred response
    - ``rejected``: Less preferred response

References:
    - DPO: https://arxiv.org/abs/2305.18290
    - IPO: https://arxiv.org/abs/2310.12036
    - KTO: https://arxiv.org/abs/2402.01306
    - ORPO: https://arxiv.org/abs/2403.07691
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import lax, random
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    lax = None
    random = None

# Flax imports
try:
    import flax
    from flax import linen as nn
    from flax.training import train_state
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    train_state = None
    optax = None


# Type aliases
Array = Any  # jnp.ndarray
PyTree = Any
PRNGKey = Any


# ==============================================================================
# Core Log-Probability Computation
# ==============================================================================

def compute_log_probs(
    logits: Array,
    labels: Array,
    attention_mask: Optional[Array] = None,
) -> Array:
    """Compute per-token log probabilities from logits.

    Args:
        logits: Model output logits of shape [batch, seq_len, vocab_size]
        labels: Token IDs of shape [batch, seq_len]
        attention_mask: Optional mask of shape [batch, seq_len], 1 for valid tokens

    Returns:
        Per-token log probabilities of shape [batch, seq_len]
    """
    # Shift logits and labels for causal LM (predict next token)
    # logits: [..., :-1, :] predicts labels: [..., 1:]
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    # Log softmax for numerical stability
    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)

    # Gather log probs for actual tokens
    # Shape: [batch, seq_len-1]
    batch_size, seq_len = shifted_labels.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]

    selected_log_probs = log_probs[batch_idx, seq_idx, shifted_labels]

    # Apply attention mask if provided
    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]
        selected_log_probs = selected_log_probs * shifted_mask

    return selected_log_probs


def compute_sequence_log_prob(
    logits: Array,
    labels: Array,
    attention_mask: Optional[Array] = None,
    average: bool = True,
) -> Array:
    """Compute total log probability for each sequence.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Token IDs [batch, seq_len]
        attention_mask: Optional mask [batch, seq_len]
        average: If True, return average log prob; otherwise sum

    Returns:
        Sequence log probabilities of shape [batch]
    """
    token_log_probs = compute_log_probs(logits, labels, attention_mask)

    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]
        seq_lengths = shifted_mask.sum(axis=-1)
        total_log_prob = token_log_probs.sum(axis=-1)

        if average:
            # Avoid division by zero
            seq_lengths = jnp.maximum(seq_lengths, 1.0)
            return total_log_prob / seq_lengths
        return total_log_prob
    else:
        if average:
            return token_log_probs.mean(axis=-1)
        return token_log_probs.sum(axis=-1)


# ==============================================================================
# Reference Model Utilities
# ==============================================================================

@dataclass
class PreferenceModelPair:
    """Container for policy and reference model parameters.

    The reference model is frozen during training. This structure allows
    efficient parameter management without duplicating the model definition.
    """
    policy_params: PyTree
    ref_params: PyTree  # Frozen, not updated during training


def create_ref_params(params: PyTree) -> PyTree:
    """Create a frozen copy of parameters for the reference model.

    Args:
        params: Current policy parameters

    Returns:
        A copy of parameters that will serve as the frozen reference
    """
    # jax.tree_util.tree_map creates a new pytree with copied arrays
    return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x), params)


def get_policy_and_ref_logits(
    apply_fn: Callable,
    policy_params: PyTree,
    ref_params: PyTree,
    input_ids: Array,
    attention_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Forward pass through both policy and reference models.

    Args:
        apply_fn: Model's apply function
        policy_params: Trainable policy parameters
        ref_params: Frozen reference parameters
        input_ids: Input token IDs
        attention_mask: Optional attention mask

    Returns:
        Tuple of (policy_logits, ref_logits)
    """
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    policy_logits = apply_fn(policy_params, **kwargs)
    ref_logits = apply_fn(ref_params, **kwargs)

    # Handle different output formats (some models return dicts/tuples)
    if isinstance(policy_logits, dict):
        policy_logits = policy_logits.get("logits", policy_logits)
    if isinstance(ref_logits, dict):
        ref_logits = ref_logits.get("logits", ref_logits)

    # Stop gradient on reference
    ref_logits = jax.lax.stop_gradient(ref_logits)

    return policy_logits, ref_logits


# ==============================================================================
# DPO Loss Function
# ==============================================================================

class DPOOutput(NamedTuple):
    """Output container for DPO loss computation."""
    loss: Array
    chosen_reward: Array
    rejected_reward: Array
    reward_margin: Array
    log_ratio_chosen: Array
    log_ratio_rejected: Array


def dpo_loss(
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
    ref_chosen_logps: Array,
    ref_rejected_logps: Array,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> DPOOutput:
    """Compute DPO (Direct Preference Optimization) loss.

    The DPO loss is:
        L = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x)
                                 - log_ref(y_w|x) + log_ref(y_l|x))))

    Which simplifies to:
        L = -log(sigmoid(beta * (pi_logratios - ref_logratios)))

    where:
        pi_logratios = log_pi(chosen) - log_pi(rejected)
        ref_logratios = log_ref(chosen) - log_ref(rejected)

    Args:
        policy_chosen_logps: Log probs of chosen under policy [batch]
        policy_rejected_logps: Log probs of rejected under policy [batch]
        ref_chosen_logps: Log probs of chosen under reference [batch]
        ref_rejected_logps: Log probs of rejected under reference [batch]
        beta: Temperature parameter (higher = more conservative)
        label_smoothing: Smoothing factor for labels (0 = no smoothing)
        reference_free: If True, ignore reference model (set ref_logratios to 0)

    Returns:
        DPOOutput with loss and diagnostic metrics
    """
    # Compute log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps

    if reference_free:
        ref_logratios = jnp.zeros_like(pi_logratios)
    else:
        ref_logratios = ref_chosen_logps - ref_rejected_logps

    # DPO "reward" margin
    logits = beta * (pi_logratios - ref_logratios)

    # Standard DPO loss with optional label smoothing
    if label_smoothing > 0:
        # Smoothed binary cross-entropy
        # L = -(1-eps)*log(sigmoid(x)) - eps*log(sigmoid(-x))
        losses = (
            -jax.nn.log_sigmoid(logits) * (1 - label_smoothing)
            - jax.nn.log_sigmoid(-logits) * label_smoothing
        )
    else:
        # Standard: -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        losses = jax.nn.softplus(-logits)

    loss = losses.mean()

    # Compute implicit rewards for logging
    # The implicit reward is: r(x,y) = beta * log(pi(y|x) / ref(y|x))
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    reward_margin = chosen_reward - rejected_reward

    return DPOOutput(
        loss=loss,
        chosen_reward=chosen_reward.mean(),
        rejected_reward=rejected_reward.mean(),
        reward_margin=reward_margin.mean(),
        log_ratio_chosen=(policy_chosen_logps - ref_chosen_logps).mean(),
        log_ratio_rejected=(policy_rejected_logps - ref_rejected_logps).mean(),
    )


# ==============================================================================
# IPO Loss Function
# ==============================================================================

class IPOOutput(NamedTuple):
    """Output container for IPO loss computation."""
    loss: Array
    reward_margin: Array


def ipo_loss(
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
    ref_chosen_logps: Array,
    ref_rejected_logps: Array,
    beta: float = 0.1,
) -> IPOOutput:
    """Compute IPO (Identity Preference Optimization) loss.

    IPO uses a squared loss instead of log-sigmoid:
        L = ((log_pi(y_w|x) - log_pi(y_l|x)) - (log_ref(y_w|x) - log_ref(y_l|x)) - 1/(2*beta))^2

    This addresses some of the optimization challenges of DPO by providing
    stronger gradients when the margin is far from the target.

    Args:
        policy_chosen_logps: Log probs of chosen under policy [batch]
        policy_rejected_logps: Log probs of rejected under policy [batch]
        ref_chosen_logps: Log probs of chosen under reference [batch]
        ref_rejected_logps: Log probs of rejected under reference [batch]
        beta: Temperature parameter

    Returns:
        IPOOutput with loss and reward margin
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # IPO target: we want the margin to equal 1/(2*beta)
    # This comes from the theoretical analysis in the IPO paper
    logits = pi_logratios - ref_logratios
    target = 1.0 / (2.0 * beta)

    # Squared error loss
    losses = (logits - target) ** 2
    loss = losses.mean()

    return IPOOutput(
        loss=loss,
        reward_margin=logits.mean(),
    )


# ==============================================================================
# KTO Loss Function
# ==============================================================================

class KTOOutput(NamedTuple):
    """Output container for KTO loss computation."""
    loss: Array
    chosen_kl: Array
    rejected_kl: Array
    chosen_reward: Array
    rejected_reward: Array


def kto_loss(
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
    ref_chosen_logps: Array,
    ref_rejected_logps: Array,
    kl_chosen: Array,
    kl_rejected: Array,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> KTOOutput:
    """Compute KTO (Kahneman-Tversky Optimization) loss.

    KTO uses asymmetric loss inspired by prospect theory:
        L_chosen = 1 - sigmoid(beta * (r_chosen - KL))
        L_rejected = 1 - sigmoid(beta * (KL - r_rejected))

    where r = log(pi/ref) is the implicit reward and KL is the
    KL divergence between policy and reference.

    This allows learning from unpaired preference data (single responses
    labeled as good or bad, without needing pairs).

    Args:
        policy_chosen_logps: Log probs of chosen under policy
        policy_rejected_logps: Log probs of rejected under policy
        ref_chosen_logps: Log probs of chosen under reference
        ref_rejected_logps: Log probs of rejected under reference
        kl_chosen: KL(policy || ref) for chosen prompts
        kl_rejected: KL(policy || ref) for rejected prompts
        beta: Temperature parameter
        desirable_weight: Weight for desirable (chosen) examples
        undesirable_weight: Weight for undesirable (rejected) examples

    Returns:
        KTOOutput with loss and diagnostic metrics
    """
    # Implicit rewards
    chosen_reward = policy_chosen_logps - ref_chosen_logps
    rejected_reward = policy_rejected_logps - ref_rejected_logps

    # KTO losses with asymmetric treatment
    # For chosen: maximize reward relative to KL baseline
    chosen_losses = 1.0 - jax.nn.sigmoid(beta * (chosen_reward - kl_chosen))

    # For rejected: minimize reward, penalize being above KL baseline
    rejected_losses = 1.0 - jax.nn.sigmoid(beta * (kl_rejected - rejected_reward))

    # Weighted combination
    loss = (
        desirable_weight * chosen_losses.mean()
        + undesirable_weight * rejected_losses.mean()
    )

    return KTOOutput(
        loss=loss,
        chosen_kl=kl_chosen.mean(),
        rejected_kl=kl_rejected.mean(),
        chosen_reward=chosen_reward.mean(),
        rejected_reward=rejected_reward.mean(),
    )


def compute_kl_divergence(
    policy_logits: Array,
    ref_logits: Array,
    labels: Array,
    attention_mask: Optional[Array] = None,
) -> Array:
    """Compute per-sequence KL divergence KL(policy || reference).

    Args:
        policy_logits: Policy model logits [batch, seq_len, vocab]
        ref_logits: Reference model logits [batch, seq_len, vocab]
        labels: Token labels [batch, seq_len]
        attention_mask: Optional mask [batch, seq_len]

    Returns:
        KL divergence per sequence [batch]
    """
    # Shift for causal LM
    policy_logits = policy_logits[:, :-1, :]
    ref_logits = ref_logits[:, :-1, :]

    # Compute log probs
    policy_log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    ref_log_probs = jax.nn.log_softmax(ref_logits, axis=-1)

    # KL = sum_x p(x) * (log p(x) - log q(x))
    # Since we want KL(policy || ref), we use policy probs
    policy_probs = jax.nn.softmax(policy_logits, axis=-1)

    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(axis=-1)

    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]
        kl_per_token = kl_per_token * shifted_mask
        return kl_per_token.sum(axis=-1) / jnp.maximum(shifted_mask.sum(axis=-1), 1.0)

    return kl_per_token.mean(axis=-1)


# ==============================================================================
# ORPO Loss Function
# ==============================================================================

class ORPOOutput(NamedTuple):
    """Output container for ORPO loss computation."""
    loss: Array
    sft_loss: Array
    odds_ratio_loss: Array
    log_odds_ratio: Array
    chosen_reward: Array
    rejected_reward: Array


def orpo_loss(
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
    chosen_labels: Array,
    chosen_logits: Array,
    attention_mask_chosen: Optional[Array] = None,
    beta: float = 0.1,
) -> ORPOOutput:
    """Compute ORPO (Odds Ratio Preference Optimization) loss.

    ORPO combines supervised fine-tuning with an odds ratio preference term:
        L = L_SFT + beta * L_OR

    where:
        L_SFT = -log(pi(chosen|x))  (standard NLL)
        L_OR = -log(sigmoid(log(odds(chosen) / odds(rejected))))

    The odds ratio formulation:
        odds(y) = p(y) / (1 - p(y))
        log_odds_ratio = log(odds(chosen)) - log(odds(rejected))
                       = log_p(chosen) - log(1-p(chosen)) - log_p(rejected) + log(1-p(rejected))

    This naturally combines reward modeling with SFT, requiring no reference model.

    Args:
        policy_chosen_logps: Log probs of chosen sequences [batch]
        policy_rejected_logps: Log probs of rejected sequences [batch]
        chosen_labels: Token labels for chosen [batch, seq_len]
        chosen_logits: Logits for chosen (for SFT loss) [batch, seq_len, vocab]
        attention_mask_chosen: Mask for chosen sequences
        beta: Weight for the odds ratio term

    Returns:
        ORPOOutput with combined loss and components
    """
    # SFT loss: standard cross-entropy on chosen responses
    # We compute per-token NLL and average
    shifted_logits = chosen_logits[:, :-1, :]
    shifted_labels = chosen_labels[:, 1:]

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)

    batch_size, seq_len = shifted_labels.shape
    batch_idx = jnp.arange(batch_size)[:, None]
    seq_idx = jnp.arange(seq_len)[None, :]

    nll = -log_probs[batch_idx, seq_idx, shifted_labels]

    if attention_mask_chosen is not None:
        shifted_mask = attention_mask_chosen[:, 1:]
        nll = nll * shifted_mask
        sft_loss = nll.sum() / jnp.maximum(shifted_mask.sum(), 1.0)
    else:
        sft_loss = nll.mean()

    # Odds ratio loss
    # For numerical stability, work in log space
    # log_odds = log_p - log(1-p)
    # Since p = exp(log_p), log(1-p) = log(1 - exp(log_p))
    # For small p: log(1-p) approx -p approx -exp(log_p)
    # For numerical stability: log(1-exp(x)) = log1p(-exp(x)) for x < 0

    def log_odds(log_p: Array) -> Array:
        """Compute log(p/(1-p)) = log(p) - log(1-p) from log(p)."""
        # log(1 - exp(log_p)) with numerical stability
        # Use log1mexp: log(1 - exp(x)) = log(-expm1(x)) for x < 0
        # Since log_p < 0 (it's a log probability), we can use:
        log_one_minus_p = jnp.log(-jnp.expm1(log_p))
        return log_p - log_one_minus_p

    log_odds_chosen = log_odds(policy_chosen_logps)
    log_odds_rejected = log_odds(policy_rejected_logps)

    log_odds_ratio = log_odds_chosen - log_odds_rejected

    # Odds ratio loss: -log(sigmoid(log_odds_ratio))
    odds_ratio_loss = jax.nn.softplus(-log_odds_ratio).mean()

    # Combined loss
    loss = sft_loss + beta * odds_ratio_loss

    return ORPOOutput(
        loss=loss,
        sft_loss=sft_loss,
        odds_ratio_loss=odds_ratio_loss,
        log_odds_ratio=log_odds_ratio.mean(),
        chosen_reward=policy_chosen_logps.mean(),
        rejected_reward=policy_rejected_logps.mean(),
    )


# ==============================================================================
# Unified Preference Loss Function
# ==============================================================================

def preference_loss(
    policy_chosen_logps: Array,
    policy_rejected_logps: Array,
    ref_chosen_logps: Array,
    ref_rejected_logps: Array,
    loss_type: str = "dpo",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    # KTO-specific
    kl_chosen: Optional[Array] = None,
    kl_rejected: Optional[Array] = None,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    # ORPO-specific
    chosen_labels: Optional[Array] = None,
    chosen_logits: Optional[Array] = None,
    attention_mask_chosen: Optional[Array] = None,
) -> Union[DPOOutput, IPOOutput, KTOOutput, ORPOOutput]:
    """Unified interface for all preference loss functions.

    Args:
        policy_chosen_logps: Log probs of chosen under policy
        policy_rejected_logps: Log probs of rejected under policy
        ref_chosen_logps: Log probs of chosen under reference
        ref_rejected_logps: Log probs of rejected under reference
        loss_type: One of "dpo", "ipo", "kto", "orpo"
        beta: Temperature parameter
        label_smoothing: Label smoothing (DPO only)
        kl_chosen: KL divergence for chosen (KTO only)
        kl_rejected: KL divergence for rejected (KTO only)
        desirable_weight: Weight for chosen (KTO only)
        undesirable_weight: Weight for rejected (KTO only)
        chosen_labels: Labels for SFT (ORPO only)
        chosen_logits: Logits for SFT (ORPO only)
        attention_mask_chosen: Mask for chosen (ORPO only)

    Returns:
        Loss output appropriate to the loss type
    """
    if loss_type == "dpo":
        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "ipo":
        return ipo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )
    elif loss_type == "kto":
        if kl_chosen is None or kl_rejected is None:
            raise ValueError("KTO requires kl_chosen and kl_rejected")
        return kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            kl_chosen,
            kl_rejected,
            beta=beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
        )
    elif loss_type == "orpo":
        if chosen_labels is None or chosen_logits is None:
            raise ValueError("ORPO requires chosen_labels and chosen_logits")
        return orpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_labels,
            chosen_logits,
            attention_mask_chosen,
            beta=beta,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==============================================================================
# Training Step Functions
# ==============================================================================

class PreferenceBatch(NamedTuple):
    """Batch of preference data for training."""
    chosen_input_ids: Array
    chosen_attention_mask: Array
    rejected_input_ids: Array
    rejected_attention_mask: Array


class PreferenceTrainState(NamedTuple):
    """Extended train state for preference training with reference model."""
    step: int
    params: PyTree
    ref_params: PyTree
    opt_state: PyTree


def create_preference_train_state(
    params: PyTree,
    optimizer: Any,  # optax optimizer
) -> PreferenceTrainState:
    """Create initial training state for preference learning.

    Args:
        params: Initial model parameters
        optimizer: Optax optimizer

    Returns:
        PreferenceTrainState with frozen reference params
    """
    ref_params = create_ref_params(params)
    opt_state = optimizer.init(params)

    return PreferenceTrainState(
        step=0,
        params=params,
        ref_params=ref_params,
        opt_state=opt_state,
    )


def make_dpo_train_step(
    apply_fn: Callable,
    optimizer: Any,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "dpo",
):
    """Create a JIT-compiled DPO/IPO training step function.

    Args:
        apply_fn: Model's apply function
        optimizer: Optax optimizer
        beta: DPO temperature
        label_smoothing: Label smoothing factor
        loss_type: "dpo" or "ipo"

    Returns:
        JIT-compiled train step function
    """

    def loss_fn(
        params: PyTree,
        ref_params: PyTree,
        batch: PreferenceBatch,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute loss and metrics for a batch."""
        # Forward pass for chosen
        chosen_logits = apply_fn(
            params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(chosen_logits, dict):
            chosen_logits = chosen_logits.get("logits", chosen_logits)

        ref_chosen_logits = apply_fn(
            ref_params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(ref_chosen_logits, dict):
            ref_chosen_logits = ref_chosen_logits.get("logits", ref_chosen_logits)
        ref_chosen_logits = jax.lax.stop_gradient(ref_chosen_logits)

        # Forward pass for rejected
        rejected_logits = apply_fn(
            params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(rejected_logits, dict):
            rejected_logits = rejected_logits.get("logits", rejected_logits)

        ref_rejected_logits = apply_fn(
            ref_params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(ref_rejected_logits, dict):
            ref_rejected_logits = ref_rejected_logits.get("logits", ref_rejected_logits)
        ref_rejected_logits = jax.lax.stop_gradient(ref_rejected_logits)

        # Compute sequence log probs
        policy_chosen_logps = compute_sequence_log_prob(
            chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        policy_rejected_logps = compute_sequence_log_prob(
            rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )
        ref_chosen_logps = compute_sequence_log_prob(
            ref_chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        ref_rejected_logps = compute_sequence_log_prob(
            ref_rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )

        # Compute loss
        if loss_type == "dpo":
            output = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
                label_smoothing=label_smoothing,
            )
            metrics = {
                "loss": output.loss,
                "chosen_reward": output.chosen_reward,
                "rejected_reward": output.rejected_reward,
                "reward_margin": output.reward_margin,
            }
        else:  # ipo
            output = ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )
            metrics = {
                "loss": output.loss,
                "reward_margin": output.reward_margin,
            }

        return output.loss, metrics

    @jax.jit
    def train_step(
        state: PreferenceTrainState,
        batch: PreferenceBatch,
    ) -> Tuple[PreferenceTrainState, Dict[str, Array]]:
        """Execute one training step."""
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state.ref_params, batch
        )

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = PreferenceTrainState(
            step=state.step + 1,
            params=new_params,
            ref_params=state.ref_params,  # Reference stays frozen
            opt_state=new_opt_state,
        )

        return new_state, metrics

    return train_step


def make_kto_train_step(
    apply_fn: Callable,
    optimizer: Any,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
):
    """Create a JIT-compiled KTO training step function.

    Args:
        apply_fn: Model's apply function
        optimizer: Optax optimizer
        beta: KTO temperature
        desirable_weight: Weight for chosen examples
        undesirable_weight: Weight for rejected examples

    Returns:
        JIT-compiled train step function
    """

    def loss_fn(
        params: PyTree,
        ref_params: PyTree,
        batch: PreferenceBatch,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute KTO loss and metrics."""
        # Forward passes
        chosen_logits = apply_fn(
            params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(chosen_logits, dict):
            chosen_logits = chosen_logits.get("logits", chosen_logits)

        rejected_logits = apply_fn(
            params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(rejected_logits, dict):
            rejected_logits = rejected_logits.get("logits", rejected_logits)

        ref_chosen_logits = apply_fn(
            ref_params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(ref_chosen_logits, dict):
            ref_chosen_logits = ref_chosen_logits.get("logits", ref_chosen_logits)
        ref_chosen_logits = jax.lax.stop_gradient(ref_chosen_logits)

        ref_rejected_logits = apply_fn(
            ref_params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(ref_rejected_logits, dict):
            ref_rejected_logits = ref_rejected_logits.get("logits", ref_rejected_logits)
        ref_rejected_logits = jax.lax.stop_gradient(ref_rejected_logits)

        # Compute log probs
        policy_chosen_logps = compute_sequence_log_prob(
            chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        policy_rejected_logps = compute_sequence_log_prob(
            rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )
        ref_chosen_logps = compute_sequence_log_prob(
            ref_chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        ref_rejected_logps = compute_sequence_log_prob(
            ref_rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )

        # Compute KL divergences
        kl_chosen = compute_kl_divergence(
            chosen_logits, ref_chosen_logits,
            batch.chosen_input_ids, batch.chosen_attention_mask
        )
        kl_rejected = compute_kl_divergence(
            rejected_logits, ref_rejected_logits,
            batch.rejected_input_ids, batch.rejected_attention_mask
        )

        # KTO loss
        output = kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            kl_chosen,
            kl_rejected,
            beta=beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
        )

        metrics = {
            "loss": output.loss,
            "chosen_kl": output.chosen_kl,
            "rejected_kl": output.rejected_kl,
            "chosen_reward": output.chosen_reward,
            "rejected_reward": output.rejected_reward,
        }

        return output.loss, metrics

    @jax.jit
    def train_step(
        state: PreferenceTrainState,
        batch: PreferenceBatch,
    ) -> Tuple[PreferenceTrainState, Dict[str, Array]]:
        """Execute one KTO training step."""
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state.ref_params, batch
        )

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = PreferenceTrainState(
            step=state.step + 1,
            params=new_params,
            ref_params=state.ref_params,
            opt_state=new_opt_state,
        )

        return new_state, metrics

    return train_step


def make_orpo_train_step(
    apply_fn: Callable,
    optimizer: Any,
    beta: float = 0.1,
):
    """Create a JIT-compiled ORPO training step function.

    ORPO does not require a reference model.

    Args:
        apply_fn: Model's apply function
        optimizer: Optax optimizer
        beta: Weight for odds ratio term

    Returns:
        JIT-compiled train step function
    """

    def loss_fn(
        params: PyTree,
        batch: PreferenceBatch,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute ORPO loss and metrics."""
        # Forward passes
        chosen_logits = apply_fn(
            params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(chosen_logits, dict):
            chosen_logits = chosen_logits.get("logits", chosen_logits)

        rejected_logits = apply_fn(
            params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(rejected_logits, dict):
            rejected_logits = rejected_logits.get("logits", rejected_logits)

        # Compute log probs
        policy_chosen_logps = compute_sequence_log_prob(
            chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        policy_rejected_logps = compute_sequence_log_prob(
            rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )

        # ORPO loss
        output = orpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            batch.chosen_input_ids,
            chosen_logits,
            batch.chosen_attention_mask,
            beta=beta,
        )

        metrics = {
            "loss": output.loss,
            "sft_loss": output.sft_loss,
            "odds_ratio_loss": output.odds_ratio_loss,
            "log_odds_ratio": output.log_odds_ratio,
            "chosen_reward": output.chosen_reward,
            "rejected_reward": output.rejected_reward,
        }

        return output.loss, metrics

    @jax.jit
    def train_step(
        params: PyTree,
        opt_state: PyTree,
        batch: PreferenceBatch,
    ) -> Tuple[PyTree, PyTree, Dict[str, Array]]:
        """Execute one ORPO training step."""
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    return train_step


# ==============================================================================
# High-Level Training Interface
# ==============================================================================

@dataclass
class PreferenceConfig:
    """Configuration for preference training."""
    loss_type: str = "dpo"  # dpo, ipo, kto, orpo
    beta: float = 0.1
    label_smoothing: float = 0.0
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    # KTO-specific
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


def create_preference_optimizer(config: PreferenceConfig) -> Any:
    """Create optimizer for preference training.

    Args:
        config: Training configuration

    Returns:
        Optax optimizer chain
    """
    if not FLAX_AVAILABLE:
        raise RuntimeError("Optax is required for preference training")

    # Learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_epochs * 1000,  # Placeholder
        end_value=config.learning_rate * 0.1,
    )

    # Optimizer chain
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )


def run_preference_training_jax(
    apply_fn: Callable,
    params: PyTree,
    train_batches: Any,  # Iterator of PreferenceBatch
    config: PreferenceConfig,
    eval_batches: Optional[Any] = None,
) -> PyTree:
    """Run preference training loop.

    Args:
        apply_fn: Model's apply function
        params: Initial model parameters
        train_batches: Iterator yielding PreferenceBatch
        config: Training configuration
        eval_batches: Optional evaluation batches

    Returns:
        Trained model parameters
    """
    if not JAX_AVAILABLE or not FLAX_AVAILABLE:
        raise RuntimeError("JAX and Flax are required for preference training")

    optimizer = create_preference_optimizer(config)

    # Create training step function based on loss type
    if config.loss_type in ("dpo", "ipo"):
        train_step = make_dpo_train_step(
            apply_fn, optimizer,
            beta=config.beta,
            label_smoothing=config.label_smoothing,
            loss_type=config.loss_type,
        )
        state = create_preference_train_state(params, optimizer)

        for epoch in range(config.num_epochs):
            for batch in train_batches:
                state, metrics = train_step(state, batch)

                if state.step % 10 == 0:
                    print(f"[jax_preference] Step {state.step}: loss={metrics['loss']:.4f}")

        return state.params

    elif config.loss_type == "kto":
        train_step = make_kto_train_step(
            apply_fn, optimizer,
            beta=config.beta,
            desirable_weight=config.desirable_weight,
            undesirable_weight=config.undesirable_weight,
        )
        state = create_preference_train_state(params, optimizer)

        for epoch in range(config.num_epochs):
            for batch in train_batches:
                state, metrics = train_step(state, batch)

                if state.step % 10 == 0:
                    print(f"[jax_preference] Step {state.step}: loss={metrics['loss']:.4f}")

        return state.params

    elif config.loss_type == "orpo":
        train_step = make_orpo_train_step(apply_fn, optimizer, beta=config.beta)
        opt_state = optimizer.init(params)

        step = 0
        for epoch in range(config.num_epochs):
            for batch in train_batches:
                params, opt_state, metrics = train_step(params, opt_state, batch)
                step += 1

                if step % 10 == 0:
                    print(f"[jax_preference] Step {step}: loss={metrics['loss']:.4f}")

        return params

    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


# ==============================================================================
# Utility Functions for Integration
# ==============================================================================

def prepare_preference_batch(
    chosen_input_ids: Array,
    chosen_attention_mask: Array,
    rejected_input_ids: Array,
    rejected_attention_mask: Array,
) -> PreferenceBatch:
    """Create a PreferenceBatch from arrays.

    This is a convenience function for creating batches from tokenized data.
    """
    return PreferenceBatch(
        chosen_input_ids=jnp.asarray(chosen_input_ids),
        chosen_attention_mask=jnp.asarray(chosen_attention_mask),
        rejected_input_ids=jnp.asarray(rejected_input_ids),
        rejected_attention_mask=jnp.asarray(rejected_attention_mask),
    )


def get_preference_loss_fn(loss_type: str) -> Callable:
    """Get the loss function for a given loss type.

    Args:
        loss_type: One of "dpo", "ipo", "kto", "orpo"

    Returns:
        The corresponding loss function
    """
    loss_fns = {
        "dpo": dpo_loss,
        "ipo": ipo_loss,
        "kto": kto_loss,
        "orpo": orpo_loss,
    }
    if loss_type not in loss_fns:
        raise ValueError(f"Unknown loss type: {loss_type}. Must be one of {list(loss_fns.keys())}")
    return loss_fns[loss_type]


# ==============================================================================
# Sharded Training Support
# ==============================================================================

def make_sharded_dpo_train_step(
    apply_fn: Callable,
    optimizer: Any,
    mesh: Mesh,
    data_pspec: PartitionSpec,
    param_pspec: PyTree,
    beta: float = 0.1,
    loss_type: str = "dpo",
):
    """Create a sharded DPO training step for distributed training.

    Args:
        apply_fn: Model's apply function
        optimizer: Optax optimizer
        mesh: JAX device mesh
        data_pspec: PartitionSpec for input data
        param_pspec: PyTree of PartitionSpecs for parameters
        beta: DPO temperature
        loss_type: "dpo" or "ipo"

    Returns:
        Sharded train step function
    """
    from jax.experimental.shard_map import shard_map

    def loss_fn(params, ref_params, batch):
        # Same as non-sharded version
        chosen_logits = apply_fn(
            params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(chosen_logits, dict):
            chosen_logits = chosen_logits.get("logits", chosen_logits)

        ref_chosen_logits = apply_fn(
            ref_params,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        if isinstance(ref_chosen_logits, dict):
            ref_chosen_logits = ref_chosen_logits.get("logits", ref_chosen_logits)
        ref_chosen_logits = jax.lax.stop_gradient(ref_chosen_logits)

        rejected_logits = apply_fn(
            params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(rejected_logits, dict):
            rejected_logits = rejected_logits.get("logits", rejected_logits)

        ref_rejected_logits = apply_fn(
            ref_params,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        if isinstance(ref_rejected_logits, dict):
            ref_rejected_logits = ref_rejected_logits.get("logits", ref_rejected_logits)
        ref_rejected_logits = jax.lax.stop_gradient(ref_rejected_logits)

        policy_chosen_logps = compute_sequence_log_prob(
            chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        policy_rejected_logps = compute_sequence_log_prob(
            rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )
        ref_chosen_logps = compute_sequence_log_prob(
            ref_chosen_logits, batch.chosen_input_ids, batch.chosen_attention_mask
        )
        ref_rejected_logps = compute_sequence_log_prob(
            ref_rejected_logits, batch.rejected_input_ids, batch.rejected_attention_mask
        )

        if loss_type == "dpo":
            output = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )
        else:
            output = ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )

        return output.loss, {"loss": output.loss}

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state.ref_params, batch
        )

        # All-reduce gradients across devices
        grads = jax.lax.pmean(grads, axis_name="data")

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = PreferenceTrainState(
            step=state.step + 1,
            params=new_params,
            ref_params=state.ref_params,
            opt_state=new_opt_state,
        )

        return new_state, metrics

    return train_step
