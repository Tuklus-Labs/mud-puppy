"""JAX/Flax implementation of GRPO (Group Relative Policy Optimization).

This module provides a pure JAX/Flax implementation of GRPO-style reinforcement
learning for language models. GRPO differs from standard PPO by using group-level
relative rewards - generating multiple completions per prompt and computing
advantages relative to the group mean.

Key features:
    - Functional JAX style with explicit state management
    - Support for advantage estimation (GAE or simple relative)
    - KL penalty to constrain policy updates
    - Reward normalization (per-batch or running)
    - Integration with jax_config.py TrainingConfig

Expected dataset format (JSON/JSONL):
    - ``prompt``: the input text to condition on
    - ``target`` (optional): reference response for reward computation
    - ``reward`` (optional): pre-computed scalar reward

Example usage:
    >>> from mud_puppy.jax_config import TrainingConfig
    >>> from mud_puppy.jax_rl import GRPOTrainer
    >>>
    >>> config = TrainingConfig(
    ...     model_name_or_path="gpt2",
    ...     dataset_path="prompts.jsonl",
    ...     output_dir="./grpo_output",
    ...     backend="jax",
    ... )
    >>> trainer = GRPOTrainer(config)
    >>> trainer.train()
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random, lax
    from jax.typing import ArrayLike
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

# Flax imports
try:
    import flax
    from flax import linen as nn
    from flax.training import train_state
    from flax.core import freeze, unfreeze
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    optax = None
    train_state = None

# For tokenization and model loading
try:
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
try:
    from .jax_config import TrainingConfig
except ImportError:
    TrainingConfig = None


# ============================================================================
# Data structures for GRPO
# ============================================================================

class GRPOBatch(NamedTuple):
    """A batch of data for GRPO training.

    Attributes:
        prompt_ids: Token IDs for prompts [batch, prompt_len]
        prompt_mask: Attention mask for prompts [batch, prompt_len]
        response_ids: Token IDs for responses [batch, num_generations, response_len]
        response_mask: Attention mask for responses [batch, num_generations, response_len]
        rewards: Scalar rewards per response [batch, num_generations]
        old_log_probs: Log probabilities under old policy [batch, num_generations, response_len]
    """
    prompt_ids: ArrayLike
    prompt_mask: ArrayLike
    response_ids: ArrayLike
    response_mask: ArrayLike
    rewards: ArrayLike
    old_log_probs: ArrayLike


class GRPORollout(NamedTuple):
    """Rollout data from generation.

    Attributes:
        sequences: Full sequences (prompt + response) [batch, num_gen, seq_len]
        log_probs: Per-token log probs [batch, num_gen, response_len]
        rewards: Rewards per sequence [batch, num_gen]
        prompt_lengths: Length of each prompt [batch]
    """
    sequences: ArrayLike
    log_probs: ArrayLike
    rewards: ArrayLike
    prompt_lengths: ArrayLike


@dataclass
class GRPOConfig:
    """Configuration specific to GRPO training.

    Attributes:
        num_generations: Number of completions per prompt for group comparison
        max_prompt_length: Maximum prompt token length
        max_response_length: Maximum response token length
        temperature: Sampling temperature for generation
        top_k: Top-k sampling parameter (0 to disable)
        top_p: Nucleus sampling parameter (1.0 to disable)
        kl_coef: Coefficient for KL penalty term
        clip_range: PPO-style clipping range for policy ratio
        normalize_rewards: Whether to normalize rewards per batch
        reward_baseline: Baseline for advantage computation ("mean", "none")
        gae_lambda: Lambda for GAE (0 for simple advantage, >0 for GAE)
        entropy_coef: Coefficient for entropy bonus
        value_coef: Coefficient for value function loss (if using critic)
    """
    num_generations: int = 4
    max_prompt_length: int = 512
    max_response_length: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    kl_coef: float = 0.1
    clip_range: float = 0.2
    normalize_rewards: bool = True
    reward_baseline: str = "mean"  # "mean", "none"
    gae_lambda: float = 0.0  # 0 for simple advantage
    entropy_coef: float = 0.01
    value_coef: float = 0.5


# ============================================================================
# Reward functions
# ============================================================================

def heuristic_reward_fn(
    responses: List[str],
    prompts: Optional[List[str]] = None,
    **kwargs,
) -> jnp.ndarray:
    """Simple heuristic reward function.

    Penalizes very short or very long responses, rewards moderate length.
    This is a placeholder - real applications should use learned reward models.

    Args:
        responses: List of generated response strings
        prompts: Optional list of prompt strings (unused in this heuristic)

    Returns:
        Array of scalar rewards [num_responses]
    """
    rewards = []
    for text in responses:
        length = len(text.split())
        if length < 4:
            rewards.append(-1.0)
        elif length > 512:
            rewards.append(-1.0)
        else:
            # Prefer moderate length responses
            rewards.append(1.0 - abs(length - 50) / 500)
    return jnp.array(rewards, dtype=jnp.float32)


# ============================================================================
# Core GRPO functions (pure JAX functional style)
# ============================================================================

def compute_log_probs(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-token log probabilities.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Token IDs [batch, seq_len]
        mask: Attention mask [batch, seq_len]

    Returns:
        Log probabilities [batch, seq_len]
    """
    # Shift for causal LM (predict next token)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = mask[:, 1:]

    # Log softmax for numerical stability
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)

    # Gather log probs for actual tokens
    batch_size, seq_len, vocab_size = shift_logits.shape
    batch_indices = jnp.arange(batch_size)[:, None]
    seq_indices = jnp.arange(seq_len)[None, :]

    token_log_probs = log_probs[batch_indices, seq_indices, shift_labels]

    # Mask out padding
    token_log_probs = token_log_probs * shift_mask

    return token_log_probs


def compute_kl_divergence(
    log_probs: jnp.ndarray,
    ref_log_probs: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL divergence between policy and reference.

    KL(policy || ref) = sum(policy * (log_policy - log_ref))

    For token-level, we use the approximation:
    KL = exp(log_policy) * (log_policy - log_ref)
       = policy * log(policy/ref)

    In practice, we use: KL = log_policy - log_ref (per token)
    and then average over the sequence.

    Args:
        log_probs: Log probs under current policy [batch, seq_len]
        ref_log_probs: Log probs under reference policy [batch, seq_len]
        mask: Attention mask [batch, seq_len]

    Returns:
        KL divergence per sequence [batch]
    """
    # Per-token KL (approximation)
    kl_per_token = log_probs - ref_log_probs

    # Mask and average over sequence
    kl_per_token = kl_per_token * mask
    sequence_lengths = jnp.sum(mask, axis=-1) + 1e-8
    kl_per_sequence = jnp.sum(kl_per_token, axis=-1) / sequence_lengths

    return kl_per_sequence


def compute_advantages(
    rewards: jnp.ndarray,
    baseline: str = "mean",
    normalize: bool = True,
) -> jnp.ndarray:
    """Compute advantages from rewards using group-relative baseline.

    For GRPO, we compute advantages relative to the group (multiple generations
    per prompt). This encourages the model to produce better responses than
    its own average.

    Args:
        rewards: Rewards [batch, num_generations]
        baseline: How to compute baseline ("mean" for group mean, "none")
        normalize: Whether to normalize advantages

    Returns:
        Advantages [batch, num_generations]
    """
    if baseline == "mean":
        # Group-relative advantage (key to GRPO)
        group_mean = jnp.mean(rewards, axis=-1, keepdims=True)
        advantages = rewards - group_mean
    else:
        advantages = rewards

    if normalize:
        # Normalize across the entire batch for stable gradients
        adv_mean = jnp.mean(advantages)
        adv_std = jnp.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    return advantages


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    For token-level RL, this computes advantages at each token position.
    For sequence-level (GRPO), use compute_advantages instead.

    Args:
        rewards: Per-step rewards [batch, seq_len]
        values: Value function estimates [batch, seq_len]
        gamma: Discount factor
        lam: GAE lambda parameter
        mask: Optional mask for valid positions

    Returns:
        Tuple of (advantages, returns) each [batch, seq_len]
    """
    batch_size, seq_len = rewards.shape

    if mask is None:
        mask = jnp.ones_like(rewards)

    # Append zero value for bootstrapping
    values_plus = jnp.concatenate([values, jnp.zeros((batch_size, 1))], axis=-1)

    def gae_step(carry, t):
        gae, next_value = carry
        reward = rewards[:, t]
        value = values[:, t]
        next_val = values_plus[:, t + 1]
        m = mask[:, t]

        delta = reward + gamma * next_val * m - value
        gae = delta + gamma * lam * gae * m

        return (gae, value), gae

    # Scan backwards through sequence
    init_gae = jnp.zeros(batch_size)
    init_value = values_plus[:, -1]
    _, advantages = lax.scan(
        gae_step,
        (init_gae, init_value),
        jnp.arange(seq_len - 1, -1, -1),
    )

    # Reverse to get correct order
    advantages = jnp.flip(advantages.T, axis=-1)
    returns = advantages + values

    return advantages, returns


def grpo_loss(
    policy_log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    ref_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    mask: jnp.ndarray,
    clip_range: float = 0.2,
    kl_coef: float = 0.1,
    entropy_coef: float = 0.01,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute GRPO loss with PPO-style clipping.

    The loss combines:
    1. Clipped policy gradient loss (PPO-style)
    2. KL penalty to stay close to reference policy
    3. Entropy bonus to encourage exploration

    Args:
        policy_log_probs: Log probs under current policy [batch, num_gen, seq_len]
        old_log_probs: Log probs under old policy (for clipping) [batch, num_gen, seq_len]
        ref_log_probs: Log probs under reference policy (for KL) [batch, num_gen, seq_len]
        advantages: Advantage estimates [batch, num_gen]
        mask: Token mask [batch, num_gen, seq_len]
        clip_range: PPO clipping range
        kl_coef: KL penalty coefficient
        entropy_coef: Entropy bonus coefficient

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Compute sequence-level log probs by summing over tokens
    seq_lengths = jnp.sum(mask, axis=-1) + 1e-8

    policy_seq_log_probs = jnp.sum(policy_log_probs * mask, axis=-1) / seq_lengths
    old_seq_log_probs = jnp.sum(old_log_probs * mask, axis=-1) / seq_lengths

    # Policy ratio
    ratio = jnp.exp(policy_seq_log_probs - old_seq_log_probs)

    # Clipped surrogate loss (PPO-style)
    # Expand advantages to match ratio shape if needed
    if advantages.ndim < ratio.ndim:
        advantages = advantages[..., None]
    elif advantages.shape != ratio.shape:
        # Advantages are [batch, num_gen], ratio is [batch, num_gen]
        pass  # Shapes already match

    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # KL penalty
    ref_seq_log_probs = jnp.sum(ref_log_probs * mask, axis=-1) / seq_lengths
    kl = jnp.mean(policy_seq_log_probs - ref_seq_log_probs)
    kl_loss = kl_coef * kl

    # Entropy bonus (encourage exploration)
    # Approximate entropy from log probs
    entropy = -jnp.mean(policy_seq_log_probs)
    entropy_loss = -entropy_coef * entropy

    total_loss = policy_loss + kl_loss + entropy_loss

    metrics = {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "kl": kl,
        "entropy": entropy,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss,
        "ratio_mean": jnp.mean(ratio),
        "ratio_std": jnp.std(ratio),
    }

    return total_loss, metrics


# ============================================================================
# Training step functions
# ============================================================================

def create_grpo_train_step(
    grpo_config: GRPOConfig,
    apply_fn: Callable,
    ref_apply_fn: Optional[Callable] = None,
) -> Callable:
    """Create a JIT-compiled GRPO training step function.

    Args:
        grpo_config: GRPO configuration
        apply_fn: Model's apply function (params, input_ids, attention_mask) -> logits
        ref_apply_fn: Reference model's apply function (or None to use frozen policy)

    Returns:
        A function (state, batch, ref_params, rng) -> (new_state, metrics)
    """

    @jax.jit
    def train_step(
        state: train_state.TrainState,
        batch: GRPOBatch,
        ref_params: Optional[Dict[str, Any]],
        rng: jax.Array,
    ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
        """Single GRPO training step.

        Args:
            state: Flax TrainState with current parameters
            batch: GRPOBatch with rollout data
            ref_params: Reference model parameters (frozen)
            rng: Random key

        Returns:
            Tuple of (updated_state, metrics_dict)
        """

        def loss_fn(params):
            # Flatten batch for model forward pass
            batch_size, num_gen, seq_len = batch.response_ids.shape

            # Concatenate prompt and response for full sequence
            # For simplicity, assume response_ids already includes prompt
            input_ids = batch.response_ids.reshape(-1, seq_len)
            attention_mask = batch.response_mask.reshape(-1, seq_len)

            # Forward pass through policy
            logits = apply_fn(params, input_ids, attention_mask)
            policy_log_probs = compute_log_probs(
                logits, input_ids, attention_mask
            )
            policy_log_probs = policy_log_probs.reshape(batch_size, num_gen, -1)

            # Reference model forward pass
            if ref_apply_fn is not None and ref_params is not None:
                ref_logits = ref_apply_fn(ref_params, input_ids, attention_mask)
            else:
                # Use old params as reference (self-play style)
                ref_logits = jax.lax.stop_gradient(
                    apply_fn(params, input_ids, attention_mask)
                )
            ref_log_probs = compute_log_probs(
                ref_logits, input_ids, attention_mask
            )
            ref_log_probs = ref_log_probs.reshape(batch_size, num_gen, -1)

            # Compute advantages from rewards
            advantages = compute_advantages(
                batch.rewards,
                baseline=grpo_config.reward_baseline,
                normalize=grpo_config.normalize_rewards,
            )

            # Reshape mask
            mask = batch.response_mask.reshape(batch_size, num_gen, -1)[:, :, 1:]

            # Compute GRPO loss
            loss, metrics = grpo_loss(
                policy_log_probs=policy_log_probs,
                old_log_probs=batch.old_log_probs[:, :, 1:] if batch.old_log_probs.shape[-1] > policy_log_probs.shape[-1] else batch.old_log_probs,
                ref_log_probs=ref_log_probs,
                advantages=advantages,
                mask=mask,
                clip_range=grpo_config.clip_range,
                kl_coef=grpo_config.kl_coef,
                entropy_coef=grpo_config.entropy_coef,
            )

            return loss, metrics

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        return new_state, metrics

    return train_step


def create_generate_fn(
    apply_fn: Callable,
    grpo_config: GRPOConfig,
    pad_token_id: int,
    eos_token_id: int,
) -> Callable:
    """Create a JIT-compiled generation function for GRPO rollouts.

    Args:
        apply_fn: Model's apply function
        grpo_config: GRPO configuration
        pad_token_id: Padding token ID
        eos_token_id: End of sequence token ID

    Returns:
        A function (params, prompt_ids, prompt_mask, rng) -> (sequences, log_probs)
    """

    @functools.partial(jax.jit, static_argnames=["num_generations"])
    def generate(
        params: Dict[str, Any],
        prompt_ids: jnp.ndarray,
        prompt_mask: jnp.ndarray,
        rng: jax.Array,
        num_generations: int = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate multiple completions per prompt.

        Args:
            params: Model parameters
            prompt_ids: Prompt token IDs [batch, prompt_len]
            prompt_mask: Prompt attention mask [batch, prompt_len]
            rng: Random key
            num_generations: Number of completions per prompt

        Returns:
            Tuple of (sequences [batch, num_gen, total_len], log_probs [batch, num_gen, response_len])
        """
        if num_generations is None:
            num_generations = grpo_config.num_generations

        batch_size, prompt_len = prompt_ids.shape
        max_new_tokens = grpo_config.max_response_length

        # Expand prompts for multiple generations
        # [batch, prompt_len] -> [batch * num_gen, prompt_len]
        expanded_prompts = jnp.repeat(prompt_ids, num_generations, axis=0)
        expanded_mask = jnp.repeat(prompt_mask, num_generations, axis=0)

        # Initialize sequences and log probs
        sequences = jnp.pad(
            expanded_prompts,
            ((0, 0), (0, max_new_tokens)),
            constant_values=pad_token_id,
        )
        all_log_probs = jnp.zeros((batch_size * num_generations, max_new_tokens))

        # Split RNG for each generation step
        rngs = random.split(rng, max_new_tokens)

        def generate_step(carry, step_rng):
            sequences, mask, position = carry

            # Get logits for current position
            logits = apply_fn(params, sequences, mask)
            next_token_logits = logits[:, position - 1, :]

            # Apply temperature
            next_token_logits = next_token_logits / grpo_config.temperature

            # Apply top-k filtering
            if grpo_config.top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(
                    next_token_logits, grpo_config.top_k
                )
                # Create mask for non-top-k tokens
                mask_value = jnp.finfo(next_token_logits.dtype).min
                next_token_logits = jnp.full_like(next_token_logits, mask_value)
                next_token_logits = next_token_logits.at[
                    jnp.arange(next_token_logits.shape[0])[:, None],
                    top_k_indices
                ].set(top_k_logits)

            # Apply top-p (nucleus) filtering
            if grpo_config.top_p < 1.0:
                sorted_logits = jnp.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > grpo_config.top_p
                # Shift to keep first token above threshold
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)

                # Get back to original order
                indices_to_remove = jnp.argsort(jnp.argsort(next_token_logits, axis=-1)[:, ::-1], axis=-1)
                indices_to_remove = jnp.take_along_axis(sorted_indices_to_remove, indices_to_remove, axis=-1)

                mask_value = jnp.finfo(next_token_logits.dtype).min
                next_token_logits = jnp.where(indices_to_remove, mask_value, next_token_logits)

            # Sample next token
            probs = jax.nn.softmax(next_token_logits, axis=-1)
            next_tokens = random.categorical(step_rng, jnp.log(probs + 1e-10), axis=-1)

            # Compute log probability of sampled token
            log_probs = jax.nn.log_softmax(next_token_logits, axis=-1)
            token_log_probs = jnp.take_along_axis(
                log_probs, next_tokens[:, None], axis=-1
            ).squeeze(-1)

            # Update sequences
            sequences = sequences.at[:, position].set(next_tokens)

            # Update attention mask
            new_mask = mask.at[:, position].set(1.0)

            return (sequences, new_mask, position + 1), token_log_probs

        # Initialize attention mask
        full_mask = jnp.pad(
            expanded_mask,
            ((0, 0), (0, max_new_tokens)),
            constant_values=0.0,
        )

        # Run generation loop
        (final_sequences, _, _), token_log_probs = lax.scan(
            generate_step,
            (sequences, full_mask, prompt_len),
            rngs,
        )

        # Reshape token_log_probs: [max_new_tokens, batch*num_gen] -> [batch*num_gen, max_new_tokens]
        token_log_probs = token_log_probs.T

        # Reshape outputs
        final_sequences = final_sequences.reshape(batch_size, num_generations, -1)
        token_log_probs = token_log_probs.reshape(batch_size, num_generations, -1)

        return final_sequences, token_log_probs

    return generate


# ============================================================================
# High-level trainer class
# ============================================================================

class GRPOTrainer:
    """High-level trainer for GRPO in JAX/Flax.

    This class manages the training loop, model loading, dataset handling,
    and checkpointing for GRPO training.

    Example:
        >>> config = TrainingConfig(...)
        >>> trainer = GRPOTrainer(config)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        grpo_config: Optional[GRPOConfig] = None,
        reward_fn: Optional[Callable] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize the GRPO trainer.

        Args:
            config: Training configuration (from jax_config.py)
            grpo_config: GRPO-specific configuration
            reward_fn: Custom reward function (responses, prompts) -> rewards
            model: Pre-loaded Flax model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for GRPOTrainer. Install with: pip install jax jaxlib")
        if not FLAX_AVAILABLE:
            raise ImportError("Flax is required for GRPOTrainer. Install with: pip install flax optax")

        self.config = config
        self.grpo_config = grpo_config or GRPOConfig()
        self.reward_fn = reward_fn or heuristic_reward_fn

        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            self._load_model_and_tokenizer()
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Initialize training state
        self._init_training_state()

        # Create JIT-compiled functions
        self._create_training_functions()

        # Initialize reference model parameters (frozen copy)
        self.ref_params = freeze(self.state.params.copy())

        # RNG state
        self.rng = random.PRNGKey(42)

        # Training metrics
        self.metrics_history: List[Dict[str, float]] = []

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from config."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers[flax]")

        print(f"[jax_rl] Loading model: {self.config.model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = FlaxAutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.get_jax_dtype() if hasattr(self.config, 'get_jax_dtype') else jnp.bfloat16,
        )

        print(f"[jax_rl] Model loaded successfully")

    def _init_training_state(self):
        """Initialize Flax TrainState."""
        # Create optimizer
        if hasattr(self.config, 'create_optimizer'):
            tx = self.config.create_optimizer()
        else:
            # Default optimizer
            tx = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adamw(learning_rate=self.config.learning_rate),
            )

        # Initialize TrainState
        self.state = train_state.TrainState.create(
            apply_fn=self._model_forward,
            params=self.model.params,
            tx=tx,
        )

    def _model_forward(
        self,
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass through the model.

        Args:
            params: Model parameters
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        outputs = self.model.module.apply(
            {"params": params},
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def _create_training_functions(self):
        """Create JIT-compiled training functions."""
        self.train_step = create_grpo_train_step(
            grpo_config=self.grpo_config,
            apply_fn=self._model_forward,
            ref_apply_fn=self._model_forward,
        )

        self.generate_fn = create_generate_fn(
            apply_fn=self._model_forward,
            grpo_config=self.grpo_config,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load the RL dataset."""
        import json

        dataset = []
        with open(self.config.dataset_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                if "prompt" in item:
                    dataset.append(item)

        print(f"[jax_rl] Loaded {len(dataset)} prompts")
        return dataset

    def _prepare_batch(
        self,
        prompts: List[str],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Tokenize and prepare a batch of prompts.

        Args:
            prompts: List of prompt strings

        Returns:
            Tuple of (prompt_ids, prompt_mask)
        """
        encodings = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.grpo_config.max_prompt_length,
            return_tensors="np",
        )

        return (
            jnp.array(encodings["input_ids"]),
            jnp.array(encodings["attention_mask"]),
        )

    def _compute_rewards(
        self,
        sequences: jnp.ndarray,
        prompt_lengths: List[int],
    ) -> jnp.ndarray:
        """Compute rewards for generated sequences.

        Args:
            sequences: Generated sequences [batch, num_gen, seq_len]
            prompt_lengths: Length of each prompt

        Returns:
            Rewards [batch, num_gen]
        """
        batch_size, num_gen, seq_len = sequences.shape

        # Decode responses
        responses = []
        for b in range(batch_size):
            prompt_len = prompt_lengths[b]
            for g in range(num_gen):
                response_ids = sequences[b, g, prompt_len:]
                response_text = self.tokenizer.decode(
                    response_ids.tolist(),
                    skip_special_tokens=True,
                )
                responses.append(response_text)

        # Compute rewards
        rewards = self.reward_fn(responses)

        # Reshape to [batch, num_gen]
        rewards = rewards.reshape(batch_size, num_gen)

        return rewards

    def train(self, num_steps: Optional[int] = None):
        """Run GRPO training.

        Args:
            num_steps: Number of training steps (defaults to dataset size * epochs)
        """
        print("[jax_rl] Starting GRPO training")
        print(f"[jax_rl] Config: {self.grpo_config}")

        dataset = self._load_dataset()

        total_steps = num_steps or (len(dataset) * self.config.num_epochs // self.config.batch_size)

        step = 0
        for epoch in range(self.config.num_epochs):
            print(f"[jax_rl] Epoch {epoch + 1}/{self.config.num_epochs}")

            # Shuffle dataset
            self.rng, shuffle_rng = random.split(self.rng)
            indices = random.permutation(shuffle_rng, len(dataset)).tolist()

            for batch_start in range(0, len(dataset), self.config.batch_size):
                if step >= total_steps:
                    break

                # Get batch
                batch_indices = indices[batch_start:batch_start + self.config.batch_size]
                batch_data = [dataset[i] for i in batch_indices]
                prompts = [item["prompt"] for item in batch_data]

                # Prepare prompts
                prompt_ids, prompt_mask = self._prepare_batch(prompts)
                prompt_lengths = [int(m.sum()) for m in prompt_mask]

                # Generate rollouts
                self.rng, gen_rng = random.split(self.rng)
                sequences, log_probs = self.generate_fn(
                    self.state.params,
                    prompt_ids,
                    prompt_mask,
                    gen_rng,
                    num_generations=self.grpo_config.num_generations,
                )

                # Compute rewards
                rewards = self._compute_rewards(sequences, prompt_lengths)

                # Create response mask (1 for response tokens, 0 for prompt/padding)
                batch_size, num_gen, seq_len = sequences.shape
                response_mask = jnp.ones((batch_size, num_gen, seq_len))
                for b, plen in enumerate(prompt_lengths):
                    response_mask = response_mask.at[b, :, :plen].set(0)

                # Create batch
                batch = GRPOBatch(
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    response_ids=sequences,
                    response_mask=response_mask,
                    rewards=rewards,
                    old_log_probs=log_probs,
                )

                # Training step
                self.rng, step_rng = random.split(self.rng)
                self.state, metrics = self.train_step(
                    self.state,
                    batch,
                    self.ref_params,
                    step_rng,
                )

                # Log metrics
                metrics_float = {k: float(v) for k, v in metrics.items()}
                self.metrics_history.append(metrics_float)

                if step % 10 == 0:
                    print(
                        f"[jax_rl] Step {step}: "
                        f"loss={metrics_float['total_loss']:.4f}, "
                        f"kl={metrics_float['kl']:.4f}, "
                        f"reward_mean={float(rewards.mean()):.4f}"
                    )

                step += 1

            # Update reference model periodically (optional)
            if (epoch + 1) % 1 == 0:  # Update every epoch
                self.ref_params = freeze(self.state.params.copy())
                print(f"[jax_rl] Updated reference model parameters")

        print("[jax_rl] Training complete!")
        self._save_model()

    def _save_model(self):
        """Save the trained model."""
        import os

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save parameters
        params_path = os.path.join(self.config.output_dir, "params.msgpack")
        with open(params_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.state.params))

        # Save tokenizer
        self.tokenizer.save_pretrained(self.config.output_dir)

        print(f"[jax_rl] Model saved to {self.config.output_dir}")

    def evaluate(
        self,
        prompts: List[str],
        num_generations: int = 4,
    ) -> Dict[str, Any]:
        """Evaluate the model on a set of prompts.

        Args:
            prompts: List of prompt strings
            num_generations: Number of generations per prompt

        Returns:
            Dictionary with evaluation results
        """
        prompt_ids, prompt_mask = self._prepare_batch(prompts)
        prompt_lengths = [int(m.sum()) for m in prompt_mask]

        self.rng, gen_rng = random.split(self.rng)
        sequences, log_probs = self.generate_fn(
            self.state.params,
            prompt_ids,
            prompt_mask,
            gen_rng,
            num_generations=num_generations,
        )

        rewards = self._compute_rewards(sequences, prompt_lengths)

        # Decode all sequences
        batch_size, num_gen, seq_len = sequences.shape
        decoded = []
        for b in range(batch_size):
            prompt_len = prompt_lengths[b]
            generations = []
            for g in range(num_gen):
                response_ids = sequences[b, g, prompt_len:]
                response_text = self.tokenizer.decode(
                    response_ids.tolist(),
                    skip_special_tokens=True,
                )
                generations.append({
                    "text": response_text,
                    "reward": float(rewards[b, g]),
                    "log_prob": float(log_probs[b, g].sum()),
                })
            decoded.append({
                "prompt": prompts[b],
                "generations": generations,
            })

        return {
            "results": decoded,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
        }


# ============================================================================
# Convenience function for CLI integration
# ============================================================================

def run_jax_grpo_training(config: TrainingConfig):
    """Run GRPO training using JAX backend.

    This function provides a simple entry point for GRPO training,
    compatible with the mud-puppy CLI.

    Args:
        config: Training configuration
    """
    if config.backend != "jax":
        print("[jax_rl] Warning: backend is not 'jax', switching to JAX backend")
        config = config.to_jax_config() if hasattr(config, 'to_jax_config') else config

    trainer = GRPOTrainer(config)
    trainer.train()


# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    "GRPOConfig",
    "GRPOBatch",
    "GRPORollout",
    "GRPOTrainer",
    "compute_log_probs",
    "compute_kl_divergence",
    "compute_advantages",
    "compute_gae",
    "grpo_loss",
    "create_grpo_train_step",
    "create_generate_fn",
    "heuristic_reward_fn",
    "run_jax_grpo_training",
]
