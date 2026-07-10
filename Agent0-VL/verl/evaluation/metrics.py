# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Metric computation utilities for Agent0-VL evaluation
Supports accuracy, verification F1, repair rate, and tool success metrics
"""

import re
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict

# Try to import math_verify for advanced math matching
try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float = 0.0
    fuzzy_accuracy: float = 0.0
    verification_precision: float = 0.0
    verification_recall: float = 0.0
    verification_f1: float = 0.0
    repair_rate: float = 0.0
    avg_steps: float = 0.0
    tool_success_rate: float = 0.0
    total_samples: int = 0
    correct_samples: int = 0
    per_dataset_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "accuracy": self.accuracy,
            "fuzzy_accuracy": self.fuzzy_accuracy,
            "verification_precision": self.verification_precision,
            "verification_recall": self.verification_recall,
            "verification_f1": self.verification_f1,
            "repair_rate": self.repair_rate,
            "avg_steps": self.avg_steps,
            "tool_success_rate": self.tool_success_rate,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "per_dataset_metrics": self.per_dataset_metrics,
        }


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    if answer is None:
        return ""
    
    answer = str(answer).strip().lower()
    
    # Remove common prefixes
    prefixes = ["the answer is", "answer:", "final answer:", "therefore,", "thus,", "so,"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove LaTeX formatting
    answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\$([^$]+)\$', r'\1', answer)
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Remove trailing punctuation
    answer = answer.rstrip('.,;:')
    
    return answer


def extract_number(text: str) -> Optional[float]:
    """
    Extract numerical value from text
    
    Args:
        text: Text potentially containing a number
        
    Returns:
        Extracted number or None
    """
    if text is None:
        return None
    
    text = str(text).strip()
    
    # Handle percentage
    if '%' in text:
        text = text.replace('%', '')
        try:
            return float(text) / 100
        except ValueError:
            pass
    
    # Handle fractions
    fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if fraction_match:
        try:
            num = float(fraction_match.group(1))
            denom = float(fraction_match.group(2))
            if denom != 0:
                return num / denom
        except ValueError:
            pass
    
    # Handle scientific notation and regular numbers
    number_match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
    if number_match:
        try:
            return float(number_match.group())
        except ValueError:
            pass
    
    return None


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Compute exact match between prediction and ground truth
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        
    Returns:
        True if exact match
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # An empty prediction (e.g. a generation error or a sample with no ground
    # truth) must never count as correct, even though "" == "".
    if not pred_norm or not gt_norm:
        return False

    return pred_norm == gt_norm


def compute_numeric_match(
    prediction: str, 
    ground_truth: str, 
    tolerance: float = 1e-5
) -> bool:
    """
    Compute numeric match with tolerance
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        tolerance: Relative tolerance for comparison
        
    Returns:
        True if numbers match within tolerance
    """
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)
    
    if pred_num is None or gt_num is None:
        return False
    
    if gt_num == 0:
        return abs(pred_num) < tolerance
    
    return abs(pred_num - gt_num) / abs(gt_num) < tolerance


def compute_math_verify_match(prediction: str, ground_truth: str) -> bool:
    """
    Use math_verify library for advanced math matching
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        
    Returns:
        True if math expressions match
    """
    if not MATH_VERIFY_AVAILABLE:
        return compute_exact_match(prediction, ground_truth)
    
    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        )
        
        ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
        prediction_truncated = prediction[-300:]  # Limit length
        
        score, _ = verify_func([ground_truth_boxed], [prediction_truncated])
        return score > 0.5
    except Exception:
        return compute_exact_match(prediction, ground_truth)


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    match_type: str = "exact"
) -> Tuple[float, int, int]:
    """
    Compute accuracy over a list of predictions
    
    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        match_type: "exact", "numeric", or "math_verify"
        
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return 0.0, 0, 0
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if match_type == "exact":
            if compute_exact_match(pred, gt):
                correct += 1
        elif match_type == "numeric":
            if compute_numeric_match(pred, gt) or compute_exact_match(pred, gt):
                correct += 1
        elif match_type == "math_verify":
            if compute_math_verify_match(pred, gt):
                correct += 1
        else:
            raise ValueError(f"Unknown match_type: {match_type}")
    
    accuracy = correct / len(predictions)
    return accuracy, correct, len(predictions)


def compute_fuzzy_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    threshold: float = 0.8
) -> Tuple[float, int, int]:
    """
    Compute fuzzy accuracy using multiple matching strategies
    
    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return 0.0, 0, 0
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        # Try exact match first
        if compute_exact_match(pred, gt):
            correct += 1
            continue
        
        # Try numeric match
        if compute_numeric_match(pred, gt):
            correct += 1
            continue
        
        # Try math_verify match
        if MATH_VERIFY_AVAILABLE and compute_math_verify_match(pred, gt):
            correct += 1
            continue
        
        # Try substring match for multiple choice
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)
        
        # Check if ground truth is a single letter (multiple choice)
        if len(gt_norm) == 1 and gt_norm.isalpha():
            # Look for the letter in prediction
            if gt_norm in pred_norm.split():
                correct += 1
                continue
    
    accuracy = correct / len(predictions)
    return accuracy, correct, len(predictions)


def compute_verification_f1(
    predicted_correct: List[bool],
    actual_correct: List[bool]
) -> Tuple[float, float, float]:
    """
    Compute verification F1 score
    
    Args:
        predicted_correct: Verifier predictions (True = step is correct)
        actual_correct: Ground truth correctness
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if len(predicted_correct) != len(actual_correct):
        raise ValueError("Predictions and actuals must have same length")
    
    if len(predicted_correct) == 0:
        return 0.0, 0.0, 0.0
    
    tp = sum(1 for p, a in zip(predicted_correct, actual_correct) if p and a)
    fp = sum(1 for p, a in zip(predicted_correct, actual_correct) if p and not a)
    fn = sum(1 for p, a in zip(predicted_correct, actual_correct) if not p and a)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_repair_rate(
    num_repairs: List[int],
    num_steps: List[int]
) -> float:
    """
    Compute repair rate (percentage of steps that required repair)
    
    Args:
        num_repairs: Number of repairs per sample
        num_steps: Number of steps per sample
        
    Returns:
        Repair rate
    """
    total_repairs = sum(num_repairs)
    total_steps = sum(num_steps)
    
    if total_steps == 0:
        return 0.0
    
    return total_repairs / total_steps


def compute_tool_success_rate(
    tool_results: List[List[Dict[str, Any]]]
) -> float:
    """
    Compute tool execution success rate
    
    Args:
        tool_results: List of tool results per sample
        
    Returns:
        Tool success rate
    """
    total_calls = 0
    successful_calls = 0
    
    for sample_results in tool_results:
        for result in sample_results:
            total_calls += 1
            if result.get("status") == "success":
                successful_calls += 1
    
    if total_calls == 0:
        return 1.0  # No tool calls means 100% success
    
    return successful_calls / total_calls


def compute_average_steps(num_steps: List[int]) -> float:
    """
    Compute average number of reasoning steps
    
    Args:
        num_steps: Number of steps per sample
        
    Returns:
        Average steps
    """
    if len(num_steps) == 0:
        return 0.0
    
    return sum(num_steps) / len(num_steps)


def aggregate_metrics(
    results: List[Dict[str, Any]],
    data_source_key: str = "data_source"
) -> EvaluationMetrics:
    """
    Aggregate metrics from evaluation results
    
    Args:
        results: List of per-sample evaluation results
        data_source_key: Key for data source in results
        
    Returns:
        Aggregated EvaluationMetrics
    """
    if len(results) == 0:
        return EvaluationMetrics()
    
    # Extract data
    predictions = [r.get("prediction", "") for r in results]
    ground_truths = [r.get("ground_truth", "") for r in results]
    num_steps = [r.get("num_steps", 1) for r in results]
    num_repairs = [r.get("num_repairs", 0) for r in results]
    tool_results = [r.get("tool_results", []) for r in results]
    
    # Compute overall metrics
    accuracy, correct, total = compute_accuracy(predictions, ground_truths, match_type="exact")
    fuzzy_acc, _, _ = compute_fuzzy_accuracy(predictions, ground_truths)
    repair_rate = compute_repair_rate(num_repairs, num_steps)
    avg_steps = compute_average_steps(num_steps)
    tool_success = compute_tool_success_rate(tool_results)
    
    # Compute verification F1 if available
    verification_precision = 0.0
    verification_recall = 0.0
    verification_f1 = 0.0
    
    if all("verification_correct" in r and "actual_correct" in r for r in results):
        predicted_correct = [r["verification_correct"] for r in results]
        actual_correct = [r["actual_correct"] for r in results]
        verification_precision, verification_recall, verification_f1 = compute_verification_f1(
            predicted_correct, actual_correct
        )
    
    # Compute per-dataset metrics
    per_dataset_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "accuracy": 0.0,
        "fuzzy_accuracy": 0.0,
        "total": 0,
        "correct": 0,
    })
    
    dataset_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        ds = r.get(data_source_key, "unknown")
        dataset_results[ds].append(r)
    
    for ds, ds_results in dataset_results.items():
        ds_predictions = [r.get("prediction", "") for r in ds_results]
        ds_ground_truths = [r.get("ground_truth", "") for r in ds_results]
        
        ds_acc, ds_correct, ds_total = compute_accuracy(ds_predictions, ds_ground_truths)
        ds_fuzzy_acc, _, _ = compute_fuzzy_accuracy(ds_predictions, ds_ground_truths)
        
        per_dataset_metrics[ds] = {
            "accuracy": ds_acc,
            "fuzzy_accuracy": ds_fuzzy_acc,
            "total": ds_total,
            "correct": ds_correct,
        }
    
    return EvaluationMetrics(
        accuracy=accuracy,
        fuzzy_accuracy=fuzzy_acc,
        verification_precision=verification_precision,
        verification_recall=verification_recall,
        verification_f1=verification_f1,
        repair_rate=repair_rate,
        avg_steps=avg_steps,
        tool_success_rate=tool_success,
        total_samples=total,
        correct_samples=correct,
        per_dataset_metrics=dict(per_dataset_metrics),
    )


def format_metrics_table(metrics: EvaluationMetrics) -> str:
    """
    Format metrics as a readable table
    
    Args:
        metrics: EvaluationMetrics object
        
    Returns:
        Formatted string table
    """
    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        f"Total Samples:        {metrics.total_samples}",
        f"Correct Samples:      {metrics.correct_samples}",
        f"Accuracy:             {metrics.accuracy * 100:.2f}%",
        f"Fuzzy Accuracy:       {metrics.fuzzy_accuracy * 100:.2f}%",
        f"Verification F1:      {metrics.verification_f1:.4f}",
        f"  - Precision:        {metrics.verification_precision:.4f}",
        f"  - Recall:           {metrics.verification_recall:.4f}",
        f"Repair Rate:          {metrics.repair_rate * 100:.2f}%",
        f"Avg Steps:            {metrics.avg_steps:.2f}",
        f"Tool Success Rate:    {metrics.tool_success_rate * 100:.2f}%",
        "-" * 60,
        "PER-DATASET BREAKDOWN:",
        "-" * 60,
    ]
    
    for ds, ds_metrics in sorted(metrics.per_dataset_metrics.items()):
        lines.append(
            f"  {ds:20s}: {ds_metrics['accuracy'] * 100:6.2f}% "
            f"({ds_metrics['correct']}/{ds_metrics['total']})"
        )
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_metrics_to_json(metrics: EvaluationMetrics, output_path: str) -> None:
    """
    Save metrics to JSON file
    
    Args:
        metrics: EvaluationMetrics object
        output_path: Path to output JSON file
    """
    with open(output_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)


def save_metrics_to_csv(metrics: EvaluationMetrics, output_path: str) -> None:
    """
    Save metrics to CSV file
    
    Args:
        metrics: EvaluationMetrics object
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write overall metrics
        writer.writerow(["Metric", "Value"])
        writer.writerow(["accuracy", f"{metrics.accuracy:.4f}"])
        writer.writerow(["fuzzy_accuracy", f"{metrics.fuzzy_accuracy:.4f}"])
        writer.writerow(["verification_f1", f"{metrics.verification_f1:.4f}"])
        writer.writerow(["repair_rate", f"{metrics.repair_rate:.4f}"])
        writer.writerow(["avg_steps", f"{metrics.avg_steps:.2f}"])
        writer.writerow(["tool_success_rate", f"{metrics.tool_success_rate:.4f}"])
        writer.writerow(["total_samples", metrics.total_samples])
        writer.writerow(["correct_samples", metrics.correct_samples])
        writer.writerow([])
        
        # Write per-dataset metrics
        writer.writerow(["Dataset", "Accuracy", "Fuzzy Accuracy", "Correct", "Total"])
        for ds, ds_metrics in sorted(metrics.per_dataset_metrics.items()):
            writer.writerow([
                ds,
                f"{ds_metrics['accuracy']:.4f}",
                f"{ds_metrics['fuzzy_accuracy']:.4f}",
                ds_metrics['correct'],
                ds_metrics['total'],
            ])
