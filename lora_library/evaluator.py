"""Adapter evaluation module for task-specific metrics.

This module provides evaluation functionality for trained LoRA adapters
with task-specific metrics like pass@k for code, accuracy for math, etc.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.

    Attributes:
        loss: Evaluation loss.
        perplexity: Perplexity score.
        accuracy: Accuracy (for applicable tasks).
        task_specific: Task-specific metrics dictionary.
    """
    loss: float = 0.0
    perplexity: float = 0.0
    accuracy: float = 0.0
    task_specific: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        result = {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
        }
        result.update(self.task_specific)
        return result


@dataclass
class EvaluationResult:
    """Result of evaluating an adapter.

    Attributes:
        adapter_name: Name of the evaluated adapter.
        task: Task type.
        metrics: Evaluation metrics.
        num_examples: Number of examples evaluated.
        evaluation_time_seconds: Time taken for evaluation.
        timestamp: Evaluation timestamp.
        comparison_metrics: Metrics comparing to base model.
    """
    adapter_name: str
    task: str
    metrics: EvaluationMetrics
    num_examples: int = 0
    evaluation_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    comparison_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "task": self.task,
            "metrics": self.metrics.to_dict(),
            "num_examples": self.num_examples,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "timestamp": self.timestamp,
            "comparison_metrics": self.comparison_metrics,
        }


class TaskEvaluator(ABC):
    """Abstract base class for task-specific evaluators."""

    task_type: str = "generic"

    @abstractmethod
    def evaluate(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        examples: List[Dict[str, Any]],
        **kwargs,
    ) -> EvaluationMetrics:
        """Evaluate the model on examples.

        Args:
            model_fn: Model forward function.
            params: Model parameters.
            examples: Evaluation examples.
            **kwargs: Additional arguments.

        Returns:
            Evaluation metrics.
        """
        pass


class CodeEvaluator(TaskEvaluator):
    """Evaluator for code generation tasks using pass@k metric."""

    task_type = "code"

    def __init__(
        self,
        k_values: List[int] = [1, 10, 100],
        num_samples: int = 200,
        timeout_seconds: int = 3,
    ):
        self.k_values = k_values
        self.num_samples = num_samples
        self.timeout_seconds = timeout_seconds

    def evaluate(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        examples: List[Dict[str, Any]],
        generate_fn: Optional[Callable] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        """Evaluate code generation with pass@k.

        Args:
            model_fn: Model forward function.
            params: Model parameters.
            examples: Evaluation examples with prompts and test cases.
            generate_fn: Function to generate completions.

        Returns:
            Metrics with pass@k scores.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for pass@k calculation")

        results = []

        for example in examples:
            prompt = example.get("prompt", "")
            test_code = example.get("test", example.get("test_list", []))

            if isinstance(test_code, list):
                test_code = "\n".join(test_code)

            # Generate samples
            if generate_fn:
                completions = generate_fn(
                    model_fn, params, prompt,
                    num_samples=min(self.num_samples, max(self.k_values)),
                )
            else:
                # Placeholder - in practice, this would use the model
                completions = [""]

            # Test each completion
            passed = []
            for completion in completions:
                full_code = prompt + completion
                is_correct = self._execute_and_test(full_code, test_code)
                passed.append(is_correct)

            results.append(passed)

        # Calculate pass@k
        task_metrics = {}
        for k in self.k_values:
            pass_at_k = self._estimate_pass_at_k(results, k)
            task_metrics[f"pass@{k}"] = pass_at_k

        return EvaluationMetrics(
            accuracy=task_metrics.get("pass@1", 0.0),
            task_specific=task_metrics,
        )

    def _execute_and_test(self, code: str, test_code: str) -> bool:
        """Execute code and run tests.

        Args:
            code: Code to execute.
            test_code: Test code to run.

        Returns:
            True if tests pass.
        """
        full_code = f"{code}\n\n{test_code}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            os.unlink(temp_path)

    def _estimate_pass_at_k(
        self,
        results: List[List[bool]],
        k: int,
    ) -> float:
        """Estimate pass@k using the unbiased estimator.

        Args:
            results: List of lists of pass/fail results per problem.
            k: k value for pass@k.

        Returns:
            pass@k estimate.
        """

        def pass_at_k(n: int, c: int, k: int) -> float:
            """Calculate pass@k for a single problem."""
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        total = 0.0
        for passed in results:
            n = len(passed)
            c = sum(passed)
            if n >= k:
                total += pass_at_k(n, c, k)

        return total / max(len(results), 1)


class MathEvaluator(TaskEvaluator):
    """Evaluator for mathematical reasoning tasks."""

    task_type = "math"

    def __init__(
        self,
        extract_answer_fn: Optional[Callable] = None,
        tolerance: float = 1e-6,
    ):
        self.extract_answer_fn = extract_answer_fn or self._default_extract_answer
        self.tolerance = tolerance

    def evaluate(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        examples: List[Dict[str, Any]],
        generate_fn: Optional[Callable] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        """Evaluate math problems with accuracy.

        Args:
            model_fn: Model forward function.
            params: Model parameters.
            examples: Evaluation examples with questions and answers.
            generate_fn: Function to generate answers.

        Returns:
            Metrics with accuracy and other scores.
        """
        correct = 0
        total = 0

        for example in examples:
            question = example.get("question", example.get("problem", ""))
            expected = example.get("answer", example.get("solution", ""))

            # Generate answer
            if generate_fn:
                generated = generate_fn(model_fn, params, question)
            else:
                generated = ""

            # Extract and compare
            extracted = self.extract_answer_fn(generated)
            expected_num = self.extract_answer_fn(expected)

            if self._compare_answers(extracted, expected_num):
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)

        return EvaluationMetrics(
            accuracy=accuracy,
            task_specific={
                "correct": correct,
                "total": total,
            },
        )

    def _default_extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text.

        Args:
            text: Text containing an answer.

        Returns:
            Extracted number or None.
        """
        # GSM8K format: #### answer
        if "####" in text:
            after_hash = text.split("####")[-1].strip()
            numbers = re.findall(r"-?\d+\.?\d*", after_hash)
            if numbers:
                try:
                    return float(numbers[0].replace(",", ""))
                except ValueError:
                    pass

        # Look for boxed answers (MATH format)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            try:
                return float(boxed_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Last number in text
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass

        return None

    def _compare_answers(
        self,
        predicted: Optional[float],
        expected: Optional[float],
    ) -> bool:
        """Compare predicted and expected answers.

        Args:
            predicted: Predicted answer.
            expected: Expected answer.

        Returns:
            True if answers match within tolerance.
        """
        if predicted is None or expected is None:
            return False

        return abs(predicted - expected) < self.tolerance


class GeneralEvaluator(TaskEvaluator):
    """General evaluator using perplexity and loss."""

    task_type = "general"

    def evaluate(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        examples: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
        **kwargs,
    ) -> EvaluationMetrics:
        """Evaluate with perplexity and loss.

        Args:
            model_fn: Model forward function.
            params: Model parameters.
            examples: Evaluation examples.
            tokenizer: Tokenizer for encoding.
            max_length: Maximum sequence length.

        Returns:
            Metrics with loss and perplexity.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX required for evaluation")

        if tokenizer is None:
            raise ValueError("Tokenizer required for general evaluation")

        total_loss = 0.0
        total_tokens = 0

        for example in examples:
            text = example.get("text", "")
            if not text:
                continue

            # Tokenize
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )

            input_ids = jnp.array(tokens["input_ids"])
            attention_mask = jnp.array(tokens["attention_mask"])

            # Forward pass
            outputs = model_fn(
                {"params": params},
                input_ids=input_ids,
                attention_mask=attention_mask,
                train=False,
            )

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            mask = attention_mask[:, 1:]

            # Calculate loss
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            label_log_probs = jnp.take_along_axis(
                log_probs,
                labels[..., None],
                axis=-1,
            ).squeeze(-1)

            loss = -(label_log_probs * mask).sum()
            num_tokens = mask.sum()

            total_loss += float(loss)
            total_tokens += int(num_tokens)

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = float(jnp.exp(avg_loss))

        return EvaluationMetrics(
            loss=avg_loss,
            perplexity=perplexity,
        )


class AdapterEvaluator:
    """Main evaluator class for adapters.

    Handles evaluation with task-specific metrics and comparison
    against base model performance.
    """

    # Registry of task evaluators
    _evaluators: Dict[str, type] = {
        "code": CodeEvaluator,
        "math": MathEvaluator,
        "general": GeneralEvaluator,
    }

    def __init__(
        self,
        model_fn: Callable,
        base_params: Dict[str, Any],
        tokenizer: Any,
    ):
        """Initialize evaluator.

        Args:
            model_fn: Model forward function.
            base_params: Base model parameters.
            tokenizer: Tokenizer.
        """
        self.model_fn = model_fn
        self.base_params = base_params
        self.tokenizer = tokenizer

        # Cache base model metrics for comparison
        self._base_metrics_cache: Dict[str, EvaluationMetrics] = {}

    @classmethod
    def register_evaluator(cls, task_type: str, evaluator_cls: type) -> None:
        """Register a custom task evaluator.

        Args:
            task_type: Task type name.
            evaluator_cls: Evaluator class.
        """
        cls._evaluators[task_type] = evaluator_cls

    def _get_evaluator(self, task: str, **kwargs) -> TaskEvaluator:
        """Get evaluator for a task type.

        Args:
            task: Task type.
            **kwargs: Evaluator arguments.

        Returns:
            TaskEvaluator instance.
        """
        evaluator_cls = self._evaluators.get(task, GeneralEvaluator)
        return evaluator_cls(**kwargs)

    def evaluate_adapter(
        self,
        adapter_name: str,
        task: str,
        adapter_params: Dict[str, Any],
        eval_examples: List[Dict[str, Any]],
        compare_to_base: bool = True,
        generate_fn: Optional[Callable] = None,
        **evaluator_kwargs,
    ) -> EvaluationResult:
        """Evaluate an adapter.

        Args:
            adapter_name: Name of the adapter.
            task: Task type.
            adapter_params: Adapter parameters.
            eval_examples: Evaluation examples.
            compare_to_base: Whether to compare to base model.
            generate_fn: Optional generation function.
            **evaluator_kwargs: Additional evaluator arguments.

        Returns:
            Evaluation result.
        """
        import time
        start_time = time.time()

        evaluator = self._get_evaluator(task, **evaluator_kwargs)

        # Evaluate adapter
        metrics = evaluator.evaluate(
            model_fn=self.model_fn,
            params=adapter_params,
            examples=eval_examples,
            tokenizer=self.tokenizer,
            generate_fn=generate_fn,
        )

        # Compare to base model
        comparison = {}
        if compare_to_base:
            cache_key = f"{task}_{len(eval_examples)}"

            if cache_key not in self._base_metrics_cache:
                base_metrics = evaluator.evaluate(
                    model_fn=self.model_fn,
                    params=self.base_params,
                    examples=eval_examples,
                    tokenizer=self.tokenizer,
                    generate_fn=generate_fn,
                )
                self._base_metrics_cache[cache_key] = base_metrics
            else:
                base_metrics = self._base_metrics_cache[cache_key]

            # Calculate improvements
            for key, value in metrics.to_dict().items():
                if key in ("loss", "perplexity"):
                    # Lower is better
                    base_val = base_metrics.to_dict().get(key, value)
                    if base_val > 0:
                        comparison[f"{key}_improvement"] = (base_val - value) / base_val
                else:
                    # Higher is better
                    base_val = base_metrics.to_dict().get(key, 0)
                    if base_val > 0:
                        comparison[f"{key}_improvement"] = (value - base_val) / base_val
                    comparison[f"{key}_base"] = base_val

        eval_time = time.time() - start_time

        return EvaluationResult(
            adapter_name=adapter_name,
            task=task,
            metrics=metrics,
            num_examples=len(eval_examples),
            evaluation_time_seconds=eval_time,
            comparison_metrics=comparison,
        )

    def evaluate_multiple(
        self,
        adapters: Dict[str, Tuple[str, Dict[str, Any]]],
        eval_examples: Dict[str, List[Dict[str, Any]]],
        **kwargs,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate multiple adapters.

        Args:
            adapters: Dict of adapter_name -> (task, params).
            eval_examples: Dict of task -> examples.
            **kwargs: Additional arguments.

        Returns:
            Dict of adapter_name -> evaluation result.
        """
        results = {}

        for adapter_name, (task, params) in adapters.items():
            examples = eval_examples.get(task, [])
            if not examples:
                print(f"[lora-library] Warning: No examples for task '{task}'")
                continue

            result = self.evaluate_adapter(
                adapter_name=adapter_name,
                task=task,
                adapter_params=params,
                eval_examples=examples,
                **kwargs,
            )
            results[adapter_name] = result

        return results

    def generate_report(
        self,
        results: Dict[str, EvaluationResult],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate an evaluation report.

        Args:
            results: Evaluation results.
            output_path: Optional path to save report.

        Returns:
            Report as string.
        """
        lines = [
            "# LoRA Adapter Evaluation Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"Total adapters evaluated: {len(results)}",
            "",
        ]

        # Group by task
        by_task: Dict[str, List[EvaluationResult]] = {}
        for result in results.values():
            if result.task not in by_task:
                by_task[result.task] = []
            by_task[result.task].append(result)

        for task, task_results in sorted(by_task.items()):
            lines.append(f"## Task: {task}")
            lines.append("")

            for result in task_results:
                lines.append(f"### {result.adapter_name}")
                lines.append("")

                # Metrics table
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for key, value in result.metrics.to_dict().items():
                    if isinstance(value, float):
                        lines.append(f"| {key} | {value:.4f} |")
                    else:
                        lines.append(f"| {key} | {value} |")

                # Comparison
                if result.comparison_metrics:
                    lines.append("")
                    lines.append("**Improvement over base:**")
                    for key, value in result.comparison_metrics.items():
                        if "improvement" in key:
                            pct = value * 100
                            sign = "+" if pct > 0 else ""
                            lines.append(f"- {key}: {sign}{pct:.2f}%")

                lines.append("")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report
