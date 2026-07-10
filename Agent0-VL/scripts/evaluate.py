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
Agent0-VL Evaluation CLI
Evaluation entry point for running benchmarks on trained models
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from verl.evaluation.agent0_evaluator import Agent0Evaluator, EvaluatorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Agent0-VL Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help="Comma-separated list of benchmarks (e.g., mathverse,mathvista,chartqa)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--enable_tools",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Enable tool execution during evaluation"
    )
    
    parser.add_argument(
        "--enable_verification",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Enable step verification during evaluation"
    )
    
    parser.add_argument(
        "--save_predictions",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Save per-sample predictions to CSV"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing benchmark data"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallelism size for vLLM"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype (bfloat16, float16, float32)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization ratio for vLLM"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments"""
    # Check model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    if not benchmarks:
        logger.error("No benchmarks specified")
        sys.exit(1)
    
    return benchmarks


def main():
    """Main evaluation entry point"""
    args = parse_args()
    benchmarks = validate_args(args)
    
    logger.info("=" * 80)
    logger.info("Agent0-VL Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Benchmarks: {', '.join(benchmarks)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Enable Tools: {args.enable_tools}")
    logger.info(f"Enable Verification: {args.enable_verification}")
    logger.info("=" * 80)
    
    try:
        # Create evaluator config
        config = EvaluatorConfig(
            model_path=args.model_path,
            batch_size=args.batch_size,
            enable_tools=args.enable_tools,
            enable_verification=args.enable_verification,
            save_per_sample_results=args.save_predictions,
            output_dir=args.output_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = Agent0Evaluator(
            model_path=args.model_path,
            config=config,
        )
        
        # Run evaluation on benchmarks
        logger.info(f"Starting evaluation on {len(benchmarks)} benchmark(s)")
        all_metrics = evaluator.evaluate_benchmarks(
            benchmark_names=benchmarks,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        for benchmark_name, metrics in sorted(all_metrics.items()):
            logger.info(f"\n{benchmark_name}:")
            logger.info(f"  Accuracy: {metrics.accuracy * 100:.2f}%")
            logger.info(f"  Fuzzy Accuracy: {metrics.fuzzy_accuracy * 100:.2f}%")
            logger.info(f"  Total Samples: {metrics.total_samples}")
            logger.info(f"  Correct: {metrics.correct_samples}")
            if hasattr(metrics, 'avg_steps'):
                logger.info(f"  Avg Steps: {metrics.avg_steps:.2f}")
            if hasattr(metrics, 'repair_rate'):
                logger.info(f"  Repair Rate: {metrics.repair_rate * 100:.2f}%")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
