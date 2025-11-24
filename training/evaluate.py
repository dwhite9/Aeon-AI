"""
Evaluation Script for Tool-Aware Fine-tuning

Measures tool selection accuracy and parameter extraction quality.
Compares base model vs fine-tuned model performance.
"""

import os
import json
import torch
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating a single example"""
    query: str
    expected_tool: str
    predicted_tool: str
    tool_correct: bool
    expected_response: str
    predicted_response: str
    metadata: Dict


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics"""
    total_examples: int
    tool_accuracy: float
    tool_precision: Dict[str, float]
    tool_recall: Dict[str, float]
    tool_f1: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    examples_by_tool: Dict[str, int]


def load_model(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    hf_token: Optional[str] = None
):
    """
    Load model with optional LoRA adapters

    Args:
        base_model_name: Base model name or path
        adapter_path: Optional path to LoRA adapters
        hf_token: HuggingFace token

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model from: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load adapters if provided
    if adapter_path:
        logger.info(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def format_instruction_for_inference(instruction: str) -> str:
    """Format instruction for model inference"""
    return f"<s>[INST] {instruction} [/INST]"


def extract_tool_from_response(response: str) -> Optional[str]:
    """
    Extract tool name from model response

    Expected format: "Tool: <tool_name>"

    Args:
        response: Model's text response

    Returns:
        Extracted tool name or None
    """
    # Try to find "Tool: <name>" pattern
    match = re.search(r'Tool:\s*(\w+)', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Check if tool name appears at start of response
    tool_names = ['rag_retrieval', 'web_search', 'code_execution', 'direct_chat']
    response_lower = response.lower().strip()

    for tool in tool_names:
        if response_lower.startswith(tool):
            return tool

    return None


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,  # Low temp for deterministic tool selection
) -> str:
    """
    Generate model response

    Args:
        model: Language model
        tokenizer: Tokenizer
        instruction: Formatted instruction
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input)
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


def evaluate_example(
    model,
    tokenizer,
    example: Dict,
    max_new_tokens: int = 256
) -> EvaluationResult:
    """
    Evaluate a single example

    Args:
        model: Language model
        tokenizer: Tokenizer
        example: Test example dict
        max_new_tokens: Max tokens to generate

    Returns:
        Evaluation result
    """
    instruction = example['instruction']
    expected_response = example['response']
    metadata = example.get('metadata', {})
    expected_tool = metadata.get('tool', 'unknown')

    # Generate prediction
    formatted_instruction = format_instruction_for_inference(instruction)
    predicted_response = generate_response(
        model, tokenizer, formatted_instruction, max_new_tokens
    )

    # Extract tool from prediction
    predicted_tool = extract_tool_from_response(predicted_response)
    if predicted_tool is None:
        predicted_tool = "none"

    # Check correctness
    tool_correct = predicted_tool == expected_tool

    return EvaluationResult(
        query=instruction.split("User Query:")[-1].strip() if "User Query:" in instruction else instruction,
        expected_tool=expected_tool,
        predicted_tool=predicted_tool,
        tool_correct=tool_correct,
        expected_response=expected_response,
        predicted_response=predicted_response,
        metadata=metadata
    )


def calculate_metrics(results: List[EvaluationResult]) -> EvaluationMetrics:
    """
    Calculate aggregated metrics from evaluation results

    Args:
        results: List of evaluation results

    Returns:
        Aggregated metrics
    """
    # Overall accuracy
    correct = sum(1 for r in results if r.tool_correct)
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    # Per-tool metrics
    tool_true_positives = defaultdict(int)
    tool_false_positives = defaultdict(int)
    tool_false_negatives = defaultdict(int)
    examples_by_tool = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for result in results:
        expected = result.expected_tool
        predicted = result.predicted_tool

        examples_by_tool[expected] += 1
        confusion[expected][predicted] += 1

        if expected == predicted:
            tool_true_positives[expected] += 1
        else:
            tool_false_negatives[expected] += 1
            tool_false_positives[predicted] += 1

    # Calculate precision, recall, F1 per tool
    precision = {}
    recall = {}
    f1 = {}

    all_tools = set(examples_by_tool.keys())

    for tool in all_tools:
        tp = tool_true_positives[tool]
        fp = tool_false_positives[tool]
        fn = tool_false_negatives[tool]

        # Precision
        if tp + fp > 0:
            precision[tool] = tp / (tp + fp)
        else:
            precision[tool] = 0.0

        # Recall
        if tp + fn > 0:
            recall[tool] = tp / (tp + fn)
        else:
            recall[tool] = 0.0

        # F1
        if precision[tool] + recall[tool] > 0:
            f1[tool] = 2 * (precision[tool] * recall[tool]) / (precision[tool] + recall[tool])
        else:
            f1[tool] = 0.0

    return EvaluationMetrics(
        total_examples=total,
        tool_accuracy=accuracy,
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1,
        confusion_matrix=dict(confusion),
        examples_by_tool=dict(examples_by_tool)
    )


def print_metrics(metrics: EvaluationMetrics, model_name: str):
    """Print evaluation metrics in a nice format"""
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {model_name}")
    print("=" * 60)

    print(f"\nOverall Tool Selection Accuracy: {metrics.tool_accuracy:.2%}")
    print(f"Total Examples: {metrics.total_examples}")

    print("\nPer-Tool Metrics:")
    print(f"{'Tool':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<8}")
    print("-" * 64)

    for tool in sorted(metrics.examples_by_tool.keys()):
        precision = metrics.tool_precision.get(tool, 0.0)
        recall = metrics.tool_recall.get(tool, 0.0)
        f1 = metrics.tool_f1.get(tool, 0.0)
        count = metrics.examples_by_tool[tool]

        print(f"{tool:<20} {precision:<12.2%} {recall:<12.2%} {f1:<12.2%} {count:<8}")

    # Macro-averaged metrics
    macro_precision = sum(metrics.tool_precision.values()) / len(metrics.tool_precision) if metrics.tool_precision else 0
    macro_recall = sum(metrics.tool_recall.values()) / len(metrics.tool_recall) if metrics.tool_recall else 0
    macro_f1 = sum(metrics.tool_f1.values()) / len(metrics.tool_f1) if metrics.tool_f1 else 0

    print("-" * 64)
    print(f"{'Macro Average':<20} {macro_precision:<12.2%} {macro_recall:<12.2%} {macro_f1:<12.2%}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    tools = sorted(metrics.examples_by_tool.keys())
    print(f"{'Actual \\ Predicted':<20} " + " ".join([f"{t:<12}" for t in tools]))
    print("-" * (20 + 13 * len(tools)))

    for actual in tools:
        row = [str(metrics.confusion_matrix[actual].get(pred, 0)) for pred in tools]
        print(f"{actual:<20} " + " ".join([f"{c:<12}" for c in row]))


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    model_name: str,
    output_dir: Optional[Path] = None
) -> Tuple[EvaluationMetrics, List[EvaluationResult]]:
    """
    Evaluate model on test dataset

    Args:
        model: Language model
        tokenizer: Tokenizer
        test_data: List of test examples
        model_name: Name for logging
        output_dir: Optional directory to save results

    Returns:
        Tuple of (metrics, detailed_results)
    """
    logger.info(f"Evaluating {model_name} on {len(test_data)} examples...")

    results = []
    for example in tqdm(test_data, desc=f"Evaluating {model_name}"):
        result = evaluate_example(model, tokenizer, example)
        results.append(result)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print_metrics(metrics, model_name)

    # Save detailed results if output dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = output_dir / f"{model_name.replace('/', '_')}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

        # Save detailed results
        results_path = output_dir / f"{model_name.replace('/', '_')}_results.json"
        with open(results_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        logger.info(f"Results saved to: {output_dir}")

    return metrics, results


def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("Tool-Aware Model Evaluation")
    print("=" * 60)

    # Configuration
    base_model = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    adapter_path = os.getenv("ADAPTER_PATH", "./output/lora_adapters")
    test_data_path = os.getenv("TEST_DATA", "./data/test.json")
    output_dir = Path(os.getenv("OUTPUT_DIR", "./evaluation_results"))
    hf_token = os.getenv("HF_TOKEN")

    # Check if adapter exists
    adapter_exists = Path(adapter_path).exists()

    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(test_data)} test examples")

    # Evaluate base model
    print("\n" + "=" * 60)
    print("EVALUATING BASE MODEL")
    print("=" * 60)

    base_model_obj, base_tokenizer = load_model(base_model, adapter_path=None, hf_token=hf_token)
    base_metrics, base_results = evaluate_model(
        base_model_obj,
        base_tokenizer,
        test_data,
        "base_model",
        output_dir
    )

    # Clean up
    del base_model_obj
    torch.cuda.empty_cache()

    # Evaluate fine-tuned model if adapters exist
    if adapter_exists:
        print("\n" + "=" * 60)
        print("EVALUATING FINE-TUNED MODEL")
        print("=" * 60)

        finetuned_model, finetuned_tokenizer = load_model(
            base_model,
            adapter_path=adapter_path,
            hf_token=hf_token
        )
        finetuned_metrics, finetuned_results = evaluate_model(
            finetuned_model,
            finetuned_tokenizer,
            test_data,
            "finetuned_model",
            output_dir
        )

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)

        improvement = finetuned_metrics.tool_accuracy - base_metrics.tool_accuracy
        improvement_pct = (improvement / base_metrics.tool_accuracy * 100) if base_metrics.tool_accuracy > 0 else 0

        print(f"\nBase Model Accuracy:       {base_metrics.tool_accuracy:.2%}")
        print(f"Fine-tuned Model Accuracy: {finetuned_metrics.tool_accuracy:.2%}")
        print(f"Improvement:               {improvement:+.2%} ({improvement_pct:+.1f}%)")

        # Per-tool comparison
        print("\nPer-Tool F1 Score Comparison:")
        print(f"{'Tool':<20} {'Base':<12} {'Fine-tuned':<12} {'Δ':<12}")
        print("-" * 56)

        for tool in sorted(base_metrics.examples_by_tool.keys()):
            base_f1 = base_metrics.tool_f1.get(tool, 0.0)
            ft_f1 = finetuned_metrics.tool_f1.get(tool, 0.0)
            delta = ft_f1 - base_f1

            print(f"{tool:<20} {base_f1:<12.2%} {ft_f1:<12.2%} {delta:+.2%}")

    else:
        logger.warning(f"No fine-tuned adapters found at: {adapter_path}")
        logger.warning("Skipping fine-tuned model evaluation")

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
