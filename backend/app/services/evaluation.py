"""Evaluation service for automatic metrics and human A/B testing."""

from typing import Dict, Any, List, Optional
from pathlib import Path


class EvaluationService:
    """
    Handles model evaluation using TSTBench-style metrics.

    Provides:
    - run_automatic_eval(model_variant, test_set_path) -> metrics JSON
    - run_human_eval_session(model_variants, prompts)

    Metrics:
    - Content Preservation: BLEU, BERTScore
    - Fluency: COLA acceptability score
    - Style: Human preference (primary signal)
    """

    def __init__(self):
        self.eval_results: Dict[str, Any] = {}

    def compute_bleu(
        self,
        candidate: str,
        reference: str
    ) -> float:
        """
        Compute BLEU score for content preservation.

        Args:
            candidate: Generated text
            reference: Ground truth text

        Returns:
            BLEU score (0-1)
        """
        # TODO: Implement using sacrebleu or similar
        return 0.0

    def compute_bertscore(
        self,
        candidate: str,
        reference: str
    ) -> float:
        """
        Compute BERTScore for semantic similarity.

        Args:
            candidate: Generated text
            reference: Ground truth text

        Returns:
            BERTScore F1 (0-1)
        """
        # TODO: Implement using bert-score library
        return 0.0

    def compute_cola_score(self, text: str) -> float:
        """
        Compute COLA acceptability score for fluency.

        Args:
            text: Text to evaluate

        Returns:
            Acceptability score (0-1)
        """
        # TODO: Implement using a fine-tuned COLA classifier
        return 0.0

    def run_automatic_eval(
        self,
        model_name: str,
        test_set_path: Path,
        generate_fn: callable
    ) -> Dict[str, Any]:
        """
        Run automatic evaluation on test set.

        Args:
            model_name: Name/ID of model being evaluated
            test_set_path: Path to test.jsonl
            generate_fn: Function that takes (subject, body) and returns reply

        Returns:
            Dictionary with aggregate metrics
        """
        results = {
            "model_name": model_name,
            "test_set": str(test_set_path),
            "metrics": {
                "content_bleu_mean": 0.0,
                "content_bleu_std": 0.0,
                "content_bertscore_mean": 0.0,
                "content_bertscore_std": 0.0,
                "fluency_cola_mean": 0.0,
                "fluency_cola_std": 0.0,
            },
            "num_samples": 0,
            "individual_scores": []
        }

        # TODO: Implement actual evaluation loop
        # 1. Load test set
        # 2. For each example:
        #    - Generate reply using generate_fn
        #    - Compute all metrics
        # 3. Aggregate results

        self.eval_results[model_name] = results
        return results

    def get_latest_results(self) -> Dict[str, Any]:
        """Get latest evaluation results for all models."""
        return self.eval_results

    def create_ab_comparison(
        self,
        prompt: Dict[str, str],
        model_a_reply: str,
        model_b_reply: str
    ) -> Dict[str, Any]:
        """
        Create A/B comparison for human evaluation.

        Args:
            prompt: {subject, body}
            model_a_reply: Reply from model A
            model_b_reply: Reply from model B

        Returns:
            Comparison dict for UI display
        """
        import random

        # Randomize order to avoid position bias
        order = random.choice(["AB", "BA"])

        return {
            "prompt": prompt,
            "option_1": model_a_reply if order == "AB" else model_b_reply,
            "option_2": model_b_reply if order == "AB" else model_a_reply,
            "order": order,
            "questions": [
                "Which better sounds like your company?",
                "Did either drop or distort any facts?"
            ]
        }
