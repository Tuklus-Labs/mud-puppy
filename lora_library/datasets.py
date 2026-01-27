"""Task-specific dataset loaders for LoRA adapter training.

This module provides dataset loading and preprocessing for various
specialized tasks including code, math, instruction following,
translation, and summarization.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

try:
    from datasets import Dataset, load_dataset, IterableDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class DatasetExample:
    """A single example from a dataset.

    Attributes:
        input_text: Input text (prompt).
        target_text: Target text (completion/response).
        metadata: Optional metadata dictionary.
    """
    input_text: str
    target_text: str
    metadata: Dict[str, Any] = None

    @property
    def full_text(self) -> str:
        """Get concatenated input + target text."""
        return f"{self.input_text}{self.target_text}"


class TaskDataset(ABC):
    """Abstract base class for task-specific datasets."""

    task_type: str = "generic"

    @abstractmethod
    def load(self) -> List[DatasetExample]:
        """Load the dataset and return examples."""
        pass

    @abstractmethod
    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        """Format a raw example into a DatasetExample."""
        pass

    def create_hf_dataset(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        num_proc: int = 4,
    ) -> Dataset:
        """Create a HuggingFace dataset from this task dataset.

        Args:
            tokenizer: Tokenizer to use.
            max_length: Maximum sequence length.
            num_proc: Number of preprocessing workers.

        Returns:
            Tokenized HuggingFace Dataset.
        """
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required")

        examples = self.load()

        # Convert to dict format
        data = {
            "text": [ex.full_text for ex in examples],
            "input": [ex.input_text for ex in examples],
            "target": [ex.target_text for ex in examples],
        }

        dataset = Dataset.from_dict(data)

        def tokenize(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text", "input", "target"],
        )


class TaskDatasetLoader:
    """Factory for loading task-specific datasets."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, task_type: str):
        """Decorator to register a dataset class for a task type."""
        def decorator(dataset_cls: type):
            cls._registry[task_type] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def load(
        cls,
        task_type: str,
        path: str,
        **kwargs,
    ) -> TaskDataset:
        """Load a dataset for a specific task type.

        Args:
            task_type: Type of task (code, math, instruction, etc.).
            path: Path to dataset.
            **kwargs: Additional arguments for the dataset loader.

        Returns:
            TaskDataset instance.
        """
        if task_type not in cls._registry:
            # Default to generic
            return GenericDataset(path, **kwargs)

        dataset_cls = cls._registry[task_type]
        return dataset_cls(path, **kwargs)

    @classmethod
    def available_tasks(cls) -> List[str]:
        """List available task types."""
        return list(cls._registry.keys())


@TaskDatasetLoader.register("generic")
class GenericDataset(TaskDataset):
    """Generic dataset loader for JSONL files."""

    task_type = "generic"

    def __init__(
        self,
        path: str,
        text_column: str = "text",
        input_column: Optional[str] = None,
        output_column: Optional[str] = None,
    ):
        self.path = path
        self.text_column = text_column
        self.input_column = input_column
        self.output_column = output_column

    def load(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self.format_example(data))

        return examples

    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        if self.input_column and self.output_column:
            return DatasetExample(
                input_text=example.get(self.input_column, ""),
                target_text=example.get(self.output_column, ""),
            )
        elif self.text_column in example:
            text = example[self.text_column]
            return DatasetExample(input_text="", target_text=text)
        else:
            # Try common column names
            if "messages" in example:
                messages = example["messages"]
                text = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in messages
                )
                return DatasetExample(input_text="", target_text=text)
            elif "prompt" in example and "completion" in example:
                return DatasetExample(
                    input_text=example["prompt"],
                    target_text=example["completion"],
                )
            elif "instruction" in example:
                input_text = example["instruction"]
                if "input" in example:
                    input_text = f"{input_text}\n{example['input']}"
                output = example.get("output", example.get("response", ""))
                return DatasetExample(input_text=input_text, target_text=output)
            else:
                # Use first string value
                for v in example.values():
                    if isinstance(v, str):
                        return DatasetExample(input_text="", target_text=v)

        return DatasetExample(input_text="", target_text="")


@TaskDatasetLoader.register("code")
class CodeDataset(TaskDataset):
    """Dataset loader for code generation tasks.

    Supports:
    - CodeSearchNet format
    - MBPP format
    - HumanEval format
    - Custom code JSONL
    """

    task_type = "code"

    def __init__(
        self,
        path: str,
        format: str = "auto",
        language: Optional[str] = None,
        include_docstring: bool = True,
        include_tests: bool = False,
    ):
        self.path = path
        self.format = format
        self.language = language
        self.include_docstring = include_docstring
        self.include_tests = include_tests

    def load(self) -> List[DatasetExample]:
        # Try to detect format
        if self.format == "auto":
            self.format = self._detect_format()

        if self.format == "codesearchnet":
            return self._load_codesearchnet()
        elif self.format == "mbpp":
            return self._load_mbpp()
        elif self.format == "humaneval":
            return self._load_humaneval()
        else:
            return self._load_jsonl()

    def _detect_format(self) -> str:
        """Auto-detect dataset format from first example."""
        try:
            with open(self.path, "r") as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)

                    if "docstring" in data and "code" in data:
                        return "codesearchnet"
                    elif "task_id" in data and "test" in data:
                        return "mbpp"
                    elif "canonical_solution" in data:
                        return "humaneval"
        except Exception:
            pass

        return "jsonl"

    def _load_codesearchnet(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_codesearchnet(data))

        return examples

    def _format_codesearchnet(self, data: Dict[str, Any]) -> DatasetExample:
        docstring = data.get("docstring", "")
        code = data.get("code", data.get("func_code_string", ""))
        language = data.get("language", self.language or "python")

        if self.include_docstring:
            input_text = f"# Language: {language}\n# Description: {docstring}\n\n"
        else:
            input_text = f"# Language: {language}\n"

        return DatasetExample(
            input_text=input_text,
            target_text=code,
            metadata={"language": language},
        )

    def _load_mbpp(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_mbpp(data))

        return examples

    def _format_mbpp(self, data: Dict[str, Any]) -> DatasetExample:
        prompt = data.get("text", data.get("prompt", ""))
        code = data.get("code", data.get("solution", ""))

        input_text = f"# Task: {prompt}\n\n"

        if self.include_tests and "test_list" in data:
            tests = "\n".join(data["test_list"])
            input_text += f"# Tests:\n# {tests}\n\n"

        return DatasetExample(
            input_text=input_text,
            target_text=code,
            metadata={"task_id": data.get("task_id", "")},
        )

    def _load_humaneval(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_humaneval(data))

        return examples

    def _format_humaneval(self, data: Dict[str, Any]) -> DatasetExample:
        prompt = data.get("prompt", "")
        solution = data.get("canonical_solution", "")

        return DatasetExample(
            input_text=prompt,
            target_text=solution,
            metadata={"task_id": data.get("task_id", "")},
        )

    def _load_jsonl(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self.format_example(data))

        return examples

    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        # Try various code-related column names
        prompt = example.get("prompt", example.get("instruction", ""))
        code = example.get("code", example.get("completion", example.get("output", "")))

        return DatasetExample(input_text=prompt, target_text=code)


@TaskDatasetLoader.register("math")
class MathDataset(TaskDataset):
    """Dataset loader for mathematical reasoning tasks.

    Supports:
    - GSM8K format
    - MATH dataset format
    - Custom math JSONL
    """

    task_type = "math"

    def __init__(
        self,
        path: str,
        format: str = "auto",
        include_chain_of_thought: bool = True,
    ):
        self.path = path
        self.format = format
        self.include_chain_of_thought = include_chain_of_thought

    def load(self) -> List[DatasetExample]:
        if self.format == "auto":
            self.format = self._detect_format()

        if self.format == "gsm8k":
            return self._load_gsm8k()
        elif self.format == "math":
            return self._load_math()
        else:
            return self._load_jsonl()

    def _detect_format(self) -> str:
        try:
            with open(self.path, "r") as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)

                    if "answer" in data and "####" in str(data.get("answer", "")):
                        return "gsm8k"
                    elif "level" in data and "type" in data:
                        return "math"
        except Exception:
            pass

        return "jsonl"

    def _load_gsm8k(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_gsm8k(data))

        return examples

    def _format_gsm8k(self, data: Dict[str, Any]) -> DatasetExample:
        question = data.get("question", "")
        answer = data.get("answer", "")

        if self.include_chain_of_thought:
            # GSM8K format: reasoning steps followed by #### final_answer
            target = answer
        else:
            # Extract just the final answer
            if "####" in answer:
                target = answer.split("####")[-1].strip()
            else:
                target = answer

        return DatasetExample(
            input_text=f"Question: {question}\n\nAnswer: ",
            target_text=target,
        )

    def _load_math(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_math(data))

        return examples

    def _format_math(self, data: Dict[str, Any]) -> DatasetExample:
        problem = data.get("problem", "")
        solution = data.get("solution", "")
        level = data.get("level", "")
        subject = data.get("type", "")

        input_text = f"Problem ({subject}, {level}): {problem}\n\nSolution: "

        return DatasetExample(
            input_text=input_text,
            target_text=solution,
            metadata={"level": level, "type": subject},
        )

    def _load_jsonl(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self.format_example(data))

        return examples

    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        question = example.get("question", example.get("problem", example.get("input", "")))
        answer = example.get("answer", example.get("solution", example.get("output", "")))

        return DatasetExample(
            input_text=f"Question: {question}\n\nAnswer: ",
            target_text=answer,
        )


@TaskDatasetLoader.register("instruction")
class InstructionDataset(TaskDataset):
    """Dataset loader for instruction-following tasks.

    Supports:
    - Alpaca format
    - ShareGPT format
    - OpenAI chat format
    - Custom instruction JSONL
    """

    task_type = "instruction"

    def __init__(
        self,
        path: str,
        format: str = "auto",
        system_prompt: Optional[str] = None,
        use_chat_template: bool = True,
    ):
        self.path = path
        self.format = format
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template

    def load(self) -> List[DatasetExample]:
        if self.format == "auto":
            self.format = self._detect_format()

        if self.format == "alpaca":
            return self._load_alpaca()
        elif self.format == "sharegpt":
            return self._load_sharegpt()
        elif self.format == "openai":
            return self._load_openai()
        else:
            return self._load_jsonl()

    def _detect_format(self) -> str:
        try:
            with open(self.path, "r") as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)

                    if "instruction" in data and "output" in data:
                        return "alpaca"
                    elif "conversations" in data:
                        return "sharegpt"
                    elif "messages" in data:
                        return "openai"
        except Exception:
            pass

        return "jsonl"

    def _load_alpaca(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_alpaca(data))

        return examples

    def _format_alpaca(self, data: Dict[str, Any]) -> DatasetExample:
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output = data.get("output", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        return DatasetExample(input_text=prompt, target_text=output)

    def _load_sharegpt(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_sharegpt(data))

        return examples

    def _format_sharegpt(self, data: Dict[str, Any]) -> DatasetExample:
        conversations = data.get("conversations", [])

        text_parts = []
        if self.system_prompt:
            text_parts.append(f"System: {self.system_prompt}")

        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))

            if role in ("human", "user"):
                text_parts.append(f"User: {content}")
            elif role in ("gpt", "assistant"):
                text_parts.append(f"Assistant: {content}")
            elif role == "system":
                text_parts.insert(0, f"System: {content}")

        # Split into input (all but last) and target (last assistant response)
        if len(text_parts) >= 2:
            input_text = "\n\n".join(text_parts[:-1]) + "\n\nAssistant: "
            target_text = text_parts[-1].replace("Assistant: ", "")
        else:
            input_text = ""
            target_text = "\n\n".join(text_parts)

        return DatasetExample(input_text=input_text, target_text=target_text)

    def _load_openai(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self._format_openai(data))

        return examples

    def _format_openai(self, data: Dict[str, Any]) -> DatasetExample:
        messages = data.get("messages", [])

        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                text_parts.insert(0, f"System: {content}")
            elif role == "user":
                text_parts.append(f"User: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")

        # Split
        if len(text_parts) >= 2:
            input_text = "\n\n".join(text_parts[:-1]) + "\n\nAssistant: "
            target_text = text_parts[-1].replace("Assistant: ", "")
        else:
            input_text = ""
            target_text = "\n\n".join(text_parts)

        return DatasetExample(input_text=input_text, target_text=target_text)

    def _load_jsonl(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self.format_example(data))

        return examples

    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        if "messages" in example:
            return self._format_openai(example)
        elif "conversations" in example:
            return self._format_sharegpt(example)
        elif "instruction" in example:
            return self._format_alpaca(example)
        else:
            prompt = example.get("prompt", example.get("input", ""))
            response = example.get("response", example.get("output", example.get("completion", "")))
            return DatasetExample(input_text=prompt, target_text=response)


@TaskDatasetLoader.register("translation")
class TranslationDataset(TaskDataset):
    """Dataset loader for translation tasks.

    Supports:
    - Opus format
    - Flores format
    - WMT format
    - Custom translation JSONL
    """

    task_type = "translation"

    def __init__(
        self,
        path: str,
        source_lang: str = "en",
        target_lang: str = "es",
        format: str = "auto",
    ):
        self.path = path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.format = format

    def load(self) -> List[DatasetExample]:
        if self.format == "auto":
            self.format = self._detect_format()

        return self._load_jsonl()

    def _detect_format(self) -> str:
        # Most translation datasets are simple parallel text
        return "jsonl"

    def _load_jsonl(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self.format_example(data))

        return examples

    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        # Try various column naming conventions
        source = (
            example.get(self.source_lang) or
            example.get("source") or
            example.get("src") or
            example.get("input") or
            ""
        )
        target = (
            example.get(self.target_lang) or
            example.get("target") or
            example.get("tgt") or
            example.get("output") or
            ""
        )

        input_text = f"Translate from {self.source_lang} to {self.target_lang}:\n\n{source}\n\nTranslation: "

        return DatasetExample(
            input_text=input_text,
            target_text=target,
            metadata={"source_lang": self.source_lang, "target_lang": self.target_lang},
        )


@TaskDatasetLoader.register("summarization")
class SummarizationDataset(TaskDataset):
    """Dataset loader for summarization tasks.

    Supports:
    - CNN/DailyMail format
    - XSum format
    - Custom summarization JSONL
    """

    task_type = "summarization"

    def __init__(
        self,
        path: str,
        format: str = "auto",
        max_source_length: int = 1024,
    ):
        self.path = path
        self.format = format
        self.max_source_length = max_source_length

    def load(self) -> List[DatasetExample]:
        if self.format == "auto":
            self.format = self._detect_format()

        return self._load_jsonl()

    def _detect_format(self) -> str:
        try:
            with open(self.path, "r") as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)

                    if "article" in data and "highlights" in data:
                        return "cnn_dailymail"
                    elif "document" in data and "summary" in data:
                        return "xsum"
        except Exception:
            pass

        return "jsonl"

    def _load_jsonl(self) -> List[DatasetExample]:
        examples = []

        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(self.format_example(data))

        return examples

    def format_example(self, example: Dict[str, Any]) -> DatasetExample:
        # CNN/DailyMail format
        if "article" in example:
            source = example["article"]
            summary = example.get("highlights", "")
        # XSum format
        elif "document" in example:
            source = example["document"]
            summary = example.get("summary", "")
        # Generic
        else:
            source = example.get("text", example.get("source", example.get("input", "")))
            summary = example.get("summary", example.get("target", example.get("output", "")))

        # Truncate source if needed
        if len(source) > self.max_source_length * 4:  # Rough char estimate
            source = source[:self.max_source_length * 4]

        input_text = f"Summarize the following text:\n\n{source}\n\nSummary: "

        return DatasetExample(input_text=input_text, target_text=summary)


def create_jax_batch_iterator(
    dataset: TaskDataset,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: int = 2048,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """Create a JAX-compatible batch iterator from a TaskDataset.

    Args:
        dataset: TaskDataset instance.
        tokenizer: Tokenizer to use.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        shuffle: Whether to shuffle.
        seed: Random seed.

    Yields:
        Batches as dictionaries of JAX arrays.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for batch iteration")

    examples = dataset.load()

    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(examples)

    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i:i + batch_size]

        texts = [ex.full_text for ex in batch_examples]

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )

        yield {
            "input_ids": jnp.array(tokenized["input_ids"]),
            "attention_mask": jnp.array(tokenized["attention_mask"]),
            "labels": jnp.array(tokenized["input_ids"]),
        }
