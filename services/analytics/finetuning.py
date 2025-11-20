"""
Embedding Fine-Tuning - Adapt embeddings to user patterns

Fine-tunes embedding model based on query patterns and user feedback.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A training example for fine-tuning"""
    query: str
    positive_documents: List[str]  # Documents that were relevant
    negative_documents: List[str]  # Documents that were not relevant
    relevance_score: float
    timestamp: datetime


class EmbeddingFineTuner:
    """
    Fine-tunes embedding model based on user interactions

    Uses query patterns and feedback to improve embedding quality.
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_model_path: str = "/models/fine-tuned-embeddings"
    ):
        """
        Initialize fine-tuner

        Args:
            base_model: Base model to fine-tune
            output_model_path: Where to save fine-tuned model
        """
        self.base_model = base_model
        self.output_model_path = output_model_path
        self.training_examples: List[TrainingExample] = []

    def collect_training_data(
        self,
        query_logs: List[Dict[str, Any]],
        min_quality_score: float = 0.7
    ) -> List[TrainingExample]:
        """
        Collect training data from query logs

        Args:
            query_logs: Historical query logs
            min_quality_score: Minimum quality score for training data

        Returns:
            List of training examples
        """
        examples = []

        for log in query_logs:
            # Only use successful queries with good quality scores
            if log.get("success") and log.get("result_quality", 0) >= min_quality_score:

                # Extract query and relevant documents
                query = log.get("query_text", "")
                tool_used = log.get("tool_used", "")

                # For RAG queries, we can extract positive/negative docs
                if tool_used == "rag_retrieval" and log.get("retrieved_documents"):
                    docs = log.get("retrieved_documents", [])

                    # High-scoring docs are positive examples
                    positive_docs = [
                        d["text"] for d in docs
                        if d.get("score", 0) >= 0.8
                    ]

                    # Lower-scoring docs are negative examples
                    negative_docs = [
                        d["text"] for d in docs
                        if 0.5 <= d.get("score", 0) < 0.8
                    ]

                    if positive_docs:
                        example = TrainingExample(
                            query=query,
                            positive_documents=positive_docs,
                            negative_documents=negative_docs,
                            relevance_score=log.get("result_quality", 0),
                            timestamp=datetime.fromisoformat(log.get("timestamp"))
                        )
                        examples.append(example)

        self.training_examples.extend(examples)
        logger.info(f"Collected {len(examples)} training examples")

        return examples

    def prepare_contrastive_pairs(
        self,
        examples: Optional[List[TrainingExample]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Prepare contrastive learning pairs

        Args:
            examples: Training examples (uses self.training_examples if None)

        Returns:
            List of (query, document, label) tuples
        """
        if examples is None:
            examples = self.training_examples

        pairs = []

        for example in examples:
            # Positive pairs (query, relevant doc, 1.0)
            for pos_doc in example.positive_documents:
                pairs.append((example.query, pos_doc, 1.0))

            # Negative pairs (query, irrelevant doc, 0.0)
            for neg_doc in example.negative_documents:
                pairs.append((example.query, neg_doc, 0.0))

        logger.info(f"Prepared {len(pairs)} contrastive pairs")
        return pairs

    def generate_synthetic_examples(
        self,
        num_examples: int = 100
    ) -> List[TrainingExample]:
        """
        Generate synthetic training examples

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of synthetic training examples
        """
        # This is a placeholder for more sophisticated generation
        # In production, you'd use techniques like:
        # - Query augmentation (paraphrasing)
        # - Document summarization
        # - Hard negative mining

        synthetic = []

        # Common query patterns for self-hosted AI
        query_templates = [
            "how to {action} {component}",
            "what is {concept}",
            "{component} {property}",
            "troubleshoot {issue}",
            "configure {setting}"
        ]

        actions = ["deploy", "configure", "optimize", "monitor", "debug"]
        components = ["vLLM", "embeddings", "RAG", "cache", "database"]
        concepts = ["semantic search", "vector database", "agent routing", "code execution"]
        properties = ["configuration", "performance", "requirements", "limitations"]
        issues = ["connection errors", "timeouts", "high latency", "out of memory"]
        settings = ["resource limits", "timeout values", "cache TTL", "batch size"]

        # Generate examples (simplified)
        # In production, you'd generate more sophisticated examples
        logger.info(f"Synthetic generation is a placeholder. Implement domain-specific logic.")

        return synthetic

    def fine_tune_model(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """
        Fine-tune the embedding model

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate

        Returns:
            Training results and metrics
        """
        logger.info("Starting model fine-tuning...")

        try:
            # Check if sentence-transformers is available
            try:
                from sentence_transformers import SentenceTransformer, InputExample, losses
                from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
                from torch.utils.data import DataLoader
            except ImportError:
                logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
                return {
                    "status": "failed",
                    "error": "sentence-transformers not installed"
                }

            # Load base model
            model = SentenceTransformer(self.base_model)

            # Prepare training data
            contrastive_pairs = self.prepare_contrastive_pairs()

            if len(contrastive_pairs) < 10:
                logger.warning("Insufficient training data. Need at least 10 examples.")
                return {
                    "status": "failed",
                    "error": "Insufficient training data",
                    "examples_count": len(contrastive_pairs)
                }

            # Convert to InputExamples
            train_examples = [
                InputExample(texts=[query, doc], label=float(label))
                for query, doc, label in contrastive_pairs
            ]

            # Create DataLoader
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size
            )

            # Define loss function (Contrastive Loss or CosineSimilarityLoss)
            train_loss = losses.CosineSimilarityLoss(model)

            # Train
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=int(len(train_dataloader) * 0.1),
                output_path=self.output_model_path,
                show_progress_bar=True
            )

            logger.info(f"Model fine-tuning complete. Saved to {self.output_model_path}")

            return {
                "status": "success",
                "model_path": self.output_model_path,
                "training_examples": len(train_examples),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    def evaluate_model(
        self,
        test_pairs: List[Tuple[str, str, float]]
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned model

        Args:
            test_pairs: Test data (query, doc, label)

        Returns:
            Evaluation metrics
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            import numpy as np

            # Load fine-tuned model
            model = SentenceTransformer(self.output_model_path)

            # Compute embeddings
            queries = [pair[0] for pair in test_pairs]
            docs = [pair[1] for pair in test_pairs]
            true_labels = [pair[2] for pair in test_pairs]

            query_embeddings = model.encode(queries)
            doc_embeddings = model.encode(docs)

            # Compute similarity scores
            similarities = np.sum(query_embeddings * doc_embeddings, axis=1)

            # Convert to binary predictions (threshold at 0.5)
            predictions = (similarities > 0.5).astype(float)

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "test_samples": len(test_pairs)
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
                "error": str(e)
            }

    def export_training_data(self, output_path: str) -> None:
        """
        Export training data for analysis

        Args:
            output_path: Where to save the data
        """
        data = []

        for example in self.training_examples:
            data.append({
                "query": example.query,
                "positive_documents": example.positive_documents,
                "negative_documents": example.negative_documents,
                "relevance_score": example.relevance_score,
                "timestamp": example.timestamp.isoformat()
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} training examples to {output_path}")
