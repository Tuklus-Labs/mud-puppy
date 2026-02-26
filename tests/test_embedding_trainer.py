"""Tests for the embedding fine-tuning method (MNRL / contrastive)."""

import json
import os

import pytest
import torch
import torch.nn.functional as F


class TestEmbeddingConfig:
    def test_embedding_method_accepted(self, tmp_path):
        from mud_puppy.config import TrainingConfig

        # Create a dummy dataset file so path validation passes
        data_path = str(tmp_path / "dummy.jsonl")
        with open(data_path, "w") as f:
            f.write('{"anchor": "a", "positive": "b"}\n')

        config = TrainingConfig(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            dataset_path=data_path,
            output_dir=str(tmp_path / "output"),
            finetuning_method="embedding",
            precision="fp32",
        )
        assert config.finetuning_method == "embedding"


class TestMNRLLoss:
    def test_mnrl_loss_positive_and_differentiable(self):
        from mud_puppy.embedding import mnrl_loss

        torch.manual_seed(42)
        # Use normalized embeddings (as they would be in the training loop)
        # with enough noise that the loss is non-trivial
        anchors_raw = torch.randn(4, 384)
        anchors = F.normalize(anchors_raw, p=2, dim=1).requires_grad_(True)
        positives_raw = anchors_raw.detach().clone() + torch.randn(4, 384) * 0.5
        positives = F.normalize(positives_raw, p=2, dim=1).requires_grad_(True)
        loss = mnrl_loss(anchors, positives, scale=20.0)
        assert loss.item() > 0
        loss.backward()
        assert anchors.grad is not None


class TestEmbeddingTrainer:
    @pytest.fixture
    def training_data(self, tmp_path):
        data_path = str(tmp_path / "train.jsonl")
        examples = [
            {
                "anchor": "what is memory caching",
                "positive": "Memory caching stores frequently accessed data in fast storage layers",
            },
            {
                "anchor": "how does vector search work",
                "positive": "Vector search uses embedding similarity to find semantically related documents",
            },
            {
                "anchor": "explain token budgets",
                "positive": "Token budgets limit how much context a cache tier can consume",
            },
            {
                "anchor": "retrieval augmented generation",
                "positive": "RAG combines document retrieval with language model generation",
            },
        ] * 5  # 20 examples
        with open(data_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        return data_path

    def test_training_runs_without_error(self, training_data, tmp_path):
        from mud_puppy.embedding import run_embedding_training
        from mud_puppy.config import TrainingConfig

        config = TrainingConfig(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            dataset_path=training_data,
            output_dir=str(tmp_path / "output"),
            finetuning_method="embedding",
            num_epochs=1,
            batch_size=4,
            learning_rate=2e-5,
            precision="fp32",
        )
        run_embedding_training(config)
        assert os.path.exists(os.path.join(config.output_dir, "config.json"))

    def test_output_loadable_as_sentence_transformer(self, training_data, tmp_path):
        from mud_puppy.embedding import run_embedding_training
        from mud_puppy.config import TrainingConfig
        from sentence_transformers import SentenceTransformer

        config = TrainingConfig(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            dataset_path=training_data,
            output_dir=str(tmp_path / "output"),
            finetuning_method="embedding",
            num_epochs=1,
            batch_size=4,
            learning_rate=2e-5,
            precision="fp32",
        )
        run_embedding_training(config)
        model = SentenceTransformer(config.output_dir)
        emb = model.encode(["test sentence"])
        assert emb.shape == (1, 384)
