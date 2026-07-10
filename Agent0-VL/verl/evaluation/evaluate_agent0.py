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
Agent0-VL Evaluation Entry Point
Provides evaluation functions for RL training integration
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union

from verl.evaluation.agent0_evaluator import Agent0Evaluator, EvaluatorConfig
from verl.evaluation.metrics import aggregate_metrics, EvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_agent0(
    model_path: str,
    config: Optional[Union[EvaluatorConfig, Dict[str, Any]]] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main evaluation function for Agent0-VL
    
    Creates an Agent0Evaluator instance and runs evaluation on provided dataset.
    
    Args:
        model_path: Path to model checkpoint
        config: Optional EvaluatorConfig or dict with configuration
        dataset: Optional list of samples to evaluate
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary containing:
            - metrics: EvaluationMetrics with accuracy, fuzzy_accuracy, repair_rate, etc.
            - results: List of per-sample evaluation results
            - config: Configuration used for evaluation
    """
    logger.info(f"Initializing Agent0Evaluator with model: {model_path}")
    
    # Initialize evaluator
    evaluator = Agent0Evaluator(
        model_path=model_path,
        config=config,
    )
    
    # If no dataset provided, return empty results
    if dataset is None or len(dataset) == 0:
        logger.warning("No dataset provided for evaluation")
        return {
            "metrics": EvaluationMetrics().to_dict(),
            "results": [],
            "config": evaluator.config.to_dict(),
        }
    
    # Run evaluation
    logger.info(f"Starting evaluation on {len(dataset)} samples")
    output = evaluator.evaluate_dataset(
        dataset=dataset,
        output_path=output_dir,
    )
    
    return output


def run_evaluation_step(
    trainer: Any,
    iteration: int,
    validation_dataset: Optional[List[Dict[str, Any]]] = None,
    benchmarks: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation step during RL training
    
    Called periodically during training to evaluate model on validation set.
    Integrates with the training loop to track performance metrics.
    
    Args:
        trainer: VERL trainer instance with model_path and config
        iteration: Current training iteration number
        validation_dataset: Optional list of validation samples
        benchmarks: Optional list of benchmark names to evaluate
        data_dir: Optional directory containing benchmark data
        
    Returns:
        Dictionary with metrics:
            - accuracy: Exact match accuracy
            - fuzzy_accuracy: Normalized answer matching accuracy
            - repair_rate: Percentage of steps requiring repair
            - avg_steps: Average reasoning steps per sample
            - tool_success_rate: Tool execution success rate
            - total_samples: Number of samples evaluated
    """
    logger.info(f"Running evaluation step at iteration {iteration}")
    
    # Get model path from trainer
    model_path = getattr(trainer, 'model_path', None)
    if model_path is None:
        # Try to get from trainer config
        if hasattr(trainer, 'config') and hasattr(trainer.config, 'model'):
            model_path = trainer.config.model.get('path')
    
    if model_path is None:
        logger.error("Could not determine model path from trainer")
        return {
            "accuracy": 0.0,
            "fuzzy_accuracy": 0.0,
            "repair_rate": 0.0,
            "avg_steps": 0.0,
            "tool_success_rate": 0.0,
            "total_samples": 0,
        }
    
    # Initialize evaluator
    evaluator = Agent0Evaluator(model_path=model_path)
    
    # Evaluate on validation dataset if provided
    if validation_dataset is not None and len(validation_dataset) > 0:
        logger.info(f"Evaluating on {len(validation_dataset)} validation samples")
        output = evaluator.evaluate_dataset(dataset=validation_dataset)
        metrics = output.get("metrics", {})
        
        return {
            "accuracy": metrics.get("accuracy", 0.0),
            "fuzzy_accuracy": metrics.get("fuzzy_accuracy", 0.0),
            "repair_rate": metrics.get("repair_rate", 0.0),
            "avg_steps": metrics.get("avg_steps", 0.0),
            "tool_success_rate": metrics.get("tool_success_rate", 0.0),
            "total_samples": metrics.get("total_samples", 0),
        }
    
    # Evaluate on benchmarks if provided
    if benchmarks is not None and len(benchmarks) > 0 and data_dir is not None:
        logger.info(f"Evaluating on benchmarks: {benchmarks}")
        all_metrics = evaluator.evaluate_benchmarks(
            benchmark_names=benchmarks,
            data_dir=data_dir,
        )
        
        # Aggregate metrics across benchmarks
        if all_metrics:
            avg_accuracy = sum(m.accuracy for m in all_metrics.values()) / len(all_metrics)
            avg_fuzzy = sum(m.fuzzy_accuracy for m in all_metrics.values()) / len(all_metrics)
            avg_repair = sum(m.repair_rate for m in all_metrics.values()) / len(all_metrics)
            avg_steps = sum(m.avg_steps for m in all_metrics.values()) / len(all_metrics)
            avg_tool = sum(m.tool_success_rate for m in all_metrics.values()) / len(all_metrics)
            total_samples = sum(m.total_samples for m in all_metrics.values())
            
            return {
                "accuracy": avg_accuracy,
                "fuzzy_accuracy": avg_fuzzy,
                "repair_rate": avg_repair,
                "avg_steps": avg_steps,
                "tool_success_rate": avg_tool,
                "total_samples": total_samples,
            }
    
    # No evaluation data provided
    logger.warning("No validation dataset or benchmarks provided for evaluation")
    return {
        "accuracy": 0.0,
        "fuzzy_accuracy": 0.0,
        "repair_rate": 0.0,
        "avg_steps": 0.0,
        "tool_success_rate": 0.0,
        "total_samples": 0,
    }


if __name__ == "__main__":
    """
    Command-line interface for evaluation
    
    Usage:
        python verl/evaluation/evaluate_agent0.py \\
            --model_path ./checkpoints/agent0/iteration_3 \\
            --data_file data/agent0/val.parquet \\
            --output_dir ./evaluation_results
    """
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(
        description="Evaluate Agent0-VL model on validation set"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to validation data (parquet or json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--enable_tools",
        type=bool,
        default=True,
        help="Enable tool execution during evaluation",
    )
    parser.add_argument(
        "--enable_verification",
        type=bool,
        default=True,
        help="Enable step verification during evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_reasoning_steps",
        type=int,
        default=8,
        help="Maximum reasoning steps per sample",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = None
    if args.data_file:
        logger.info(f"Loading data from {args.data_file}")
        if args.data_file.endswith(".parquet"):
            df = pd.read_parquet(args.data_file)
            dataset = df.to_dict("records")
        elif args.data_file.endswith(".json"):
            with open(args.data_file, "r") as f:
                dataset = json.load(f)
        else:
            logger.error(f"Unsupported file format: {args.data_file}")
            exit(1)
    
    # Create config
    config = EvaluatorConfig(
        enable_tools=args.enable_tools,
        enable_verification=args.enable_verification,
        batch_size=args.batch_size,
        max_reasoning_steps=args.max_reasoning_steps,
        output_dir=args.output_dir,
    )
    
    # Run evaluation
    results = evaluate_agent0(
        model_path=args.model_path,
        config=config,
        dataset=dataset,
        output_dir=os.path.join(args.output_dir, "results.json"),
    )
    
    # Print summary
    metrics = results.get("metrics", {})
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics.get('accuracy', 0.0):.2%}")
    logger.info(f"Fuzzy Accuracy: {metrics.get('fuzzy_accuracy', 0.0):.2%}")
    logger.info(f"Repair Rate: {metrics.get('repair_rate', 0.0):.2%}")
    logger.info(f"Avg Steps: {metrics.get('avg_steps', 0.0):.1f}")
    logger.info(f"Tool Success Rate: {metrics.get('tool_success_rate', 0.0):.2%}")
    logger.info(f"Total Samples: {metrics.get('total_samples', 0)}")
    logger.info("=" * 60)
