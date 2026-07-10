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
Agent0-VL Main Evaluator
Comprehensive evaluation class for running SERC cycle on benchmarks
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict
from tqdm import tqdm



from verl.evaluation.metrics import (
    EvaluationMetrics,
    aggregate_metrics,
    format_metrics_table,
    save_metrics_to_json,
    save_metrics_to_csv,
    compute_exact_match,
    compute_fuzzy_accuracy,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for Agent0Evaluator"""
    # Model settings
    model_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.8
    
    # Generation settings
    max_reasoning_steps: int = 8
    response_length: int = 2048
    prompt_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # SERC settings
    enable_tools: bool = True
    enable_verification: bool = True
    enable_self_repair: bool = True
    repair_threshold: float = 0.7
    
    # Evaluation settings
    batch_size: int = 16
    num_workers: int = 4
    save_intermediate: bool = True
    intermediate_save_interval: int = 100
    
    # Output settings
    output_dir: str = "./evaluation_results"
    save_per_sample_results: bool = True

    # Testing only: when True, missing vLLM does not raise and generation
    # returns an empty placeholder response (never the ground truth).
    mock_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_reasoning_steps": self.max_reasoning_steps,
            "response_length": self.response_length,
            "prompt_length": self.prompt_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_tools": self.enable_tools,
            "enable_verification": self.enable_verification,
            "enable_self_repair": self.enable_self_repair,
            "repair_threshold": self.repair_threshold,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "save_intermediate": self.save_intermediate,
            "intermediate_save_interval": self.intermediate_save_interval,
            "output_dir": self.output_dir,
            "save_per_sample_results": self.save_per_sample_results,
            "mock_mode": self.mock_mode,
        }


class Agent0Evaluator:
    """
    Main evaluation class for Agent0-VL
    
    Supports:
    - Full SERC cycle evaluation
    - Batch processing with vLLM
    - Multiple benchmark evaluation
    - Intermediate result saving for resume
    - Detailed per-sample logging
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Union[EvaluatorConfig, Dict[str, Any]]] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        """
        Initialize Agent0Evaluator
        
        Args:
            model_path: Path to model checkpoint
            config: Evaluator configuration
            tokenizer: Optional pre-loaded tokenizer
            model: Optional pre-loaded model (for testing)
        """
        # Handle config
        if config is None:
            self.config = EvaluatorConfig(model_path=model_path)
        elif isinstance(config, dict):
            self.config = EvaluatorConfig(**config)
            self.config.model_path = model_path
        else:
            self.config = config
            self.config.model_path = model_path
        
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model
        self.inference_engine = None
        self.tool_registry = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize if model not provided
        if model is None and tokenizer is None:
            self._initialize_model()
        
        logger.info(f"Agent0Evaluator initialized with model: {model_path}")
    
    def _initialize_model(self):
        """Initialize model, tokenizer, and tools"""
        try:
            from transformers import AutoTokenizer, AutoConfig
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model config
            model_config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            
            # Calculate max model length
            max_model_len = (
                self.config.prompt_length + 
                self.config.max_reasoning_steps * self.config.response_length * 3
            )
            
            if hasattr(model_config, 'max_position_embeddings'):
                max_model_len = min(max_model_len, model_config.max_position_embeddings)
            
            logger.info(f"Initializing vLLM with max_model_len={max_model_len}")
            
            # Initialize vLLM
            self.inference_engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                seed=42,
            )
            
            # Initialize sampling params
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_tokens=self.config.response_length,
            )
            
            logger.info("Model initialized successfully")
            
        except ImportError as e:
            if self.config.mock_mode:
                logger.warning(f"Could not import vLLM: {e}. Running in mock mode (predictions will be empty).")
                self.inference_engine = None
            else:
                raise RuntimeError(
                    f"vLLM is required for evaluation but could not be imported: {e}. "
                    "Install vllm, or set mock_mode=True for a dry run (no real predictions)."
                ) from e
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
        
        # Initialize tools if enabled
        if self.config.enable_tools:
            try:
                from verl.utils.tools.tool_registry import get_global_tool_registry
                self.tool_registry = get_global_tool_registry()
                logger.info("Tool registry initialized")
            except ImportError:
                logger.warning("Could not import tool registry")
                self.tool_registry = None
    
    def _build_prompt(
        self,
        question: str,
        image_path: Optional[str] = None,
        has_image: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build prompt for evaluation

        Args:
            question: Question text
            image_path: Optional path to image on disk
            has_image: True when an image (PIL or path) will be fed to the model;
                a vision placeholder must then be emitted in the prompt so the
                number of ``<image>`` tokens matches ``multi_modal_data``.
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        # Aligned with the training system prompt (scripts/prompt.txt): reason
        # step by step, optionally run Python in fenced code blocks executed by a
        # sandbox, and give the final answer in \boxed{...}.
        if system_prompt is None:
            system_prompt = (
                "You are a vision-language reasoning agent. Solve the problem step "
                "by step. You may optionally write Python code to manipulate the "
                "image (crop, resize, adjust contrast) or to perform calculations "
                "that support your reasoning.\n\n"
                "- Wrap every Python snippet in a fenced block:\n"
                "  ```python\n  # your code\n  ```\n"
                "  The code runs in a sandbox and its output is returned to you.\n"
                "- When you are finished, put the final answer inside \\boxed{...}."
            )

        # Emit exactly one image content block iff an image will actually be fed
        # to the model. Generation always uses the resolved PIL image (see
        # evaluate_sample), so `has_image` is the single source of truth for the
        # placeholder count — keying off image_path alone could emit a
        # placeholder with no matching image (e.g. path exists but PIL load
        # failed) and desync the vision token count.
        if has_image:
            image_block = {"type": "image"}
            if isinstance(image_path, str) and image_path:
                image_block["image"] = image_path
            user_content = [image_block, {"type": "text", "text": question}]
        else:
            user_content = question

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        if self.tokenizer is not None and hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{system_prompt}\n\nUser: {question}\n\nAssistant:"
        
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """
        Extract final answer from model response
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer
        """
        import re
        
        # Look for FINAL_ANSWER marker
        pattern = r'FINAL_ANSWER:\s*(.+?)(?:\n|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        # Look for boxed answer
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, response)
        
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        # Return last line as fallback
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    def _extract_confidence(self, response: str) -> float:
        """
        Extract confidence from model response
        
        Args:
            response: Model response text
            
        Returns:
            Confidence score (0-1)
        """
        import re
        
        pattern = r'CONFIDENCE:\s*([\d.]+)'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        if matches:
            try:
                conf = float(matches[-1])
                return max(0.0, min(1.0, conf))
            except ValueError:
                pass
        
        return 0.5  # Default confidence
    
    def _count_steps(self, response: str) -> int:
        """
        Count reasoning steps in response
        
        Args:
            response: Model response text
            
        Returns:
            Number of reasoning steps
        """
        import re
        
        think_pattern = r'<think>.*?</think>'
        matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)
        
        return max(1, len(matches))
    
    def _count_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the response.

        Counts the Python code blocks that ``_generate_with_tools`` actually
        executes, so the recorded tool calls match what was run (the model and
        the executor both use fenced ```python``` blocks, not a JSON protocol).

        Args:
            response: Model response text

        Returns:
            List of tool call dictionaries
        """
        import re

        code_blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n?(.*?)```", response, re.DOTALL)
        return [
            {"tool_name": "PythonExec", "tool_input": cb.strip(), "status": "executed"}
            for cb in code_blocks if cb.strip()
        ]
    
    def _generate_with_tools(self, prompt: str, pil_image=None) -> str:
        """Multi-turn tool-integrated generation.

        Generates up to ``max_reasoning_steps`` rounds; after each round any
        ```python``` code blocks are executed in the sandbox and their output
        is appended to the running context. Stops when a final answer
        (FINAL_ANSWER / \\boxed{}) appears or no tool call is made.
        """
        import re

        def _vllm_generate(text_prompt: str) -> str:
            if pil_image is not None:
                request = {"prompt": text_prompt, "multi_modal_data": {"image": [pil_image]}}
            else:
                request = {"prompt": text_prompt}
            outputs = self.inference_engine.generate([request], self.sampling_params)
            return outputs[0].outputs[0].text

        full_response = ""
        context = prompt

        for _ in range(max(1, self.config.max_reasoning_steps)):
            segment = _vllm_generate(context)
            full_response += segment
            context += segment

            has_final = bool(re.search(r"FINAL_ANSWER:", segment, re.IGNORECASE)) or "\\boxed{" in segment
            code_blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n?(.*?)```", segment, re.DOTALL)

            if has_final or not code_blocks or not self.config.enable_tools:
                break

            # Execute the code blocks in the sandbox
            try:
                import asyncio
                from sandbox import get_parallel_sandbox
                parallel_sandbox = get_parallel_sandbox()
                ok, out, err = asyncio.run(parallel_sandbox([cb.strip() for cb in code_blocks]))
                obs_text = "\n[Code Execution Result]\n"
                for success, stdout, stderr in zip(ok, out, err):
                    if stderr:
                        obs_text += f"Error: {str(stderr)[-512:]}\n"
                    elif stdout:
                        obs_text += f"Output: {str(stdout)[:512]}\n"
                    else:
                        obs_text += "No output\n"
            except Exception as e:
                obs_text = f"\n[Code Execution Result]\nError: sandbox unavailable ({e})\n"

            full_response += obs_text
            context += obs_text

        return full_response

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run full SERC cycle on single sample
        
        Args:
            sample: Sample dictionary with 'question', 'image_path', 'ground_truth'
            sample_idx: Sample index for logging
            
        Returns:
            Evaluation result dictionary
        """
        start_time = time.time()
        
        # Extract sample data
        question = sample.get("question", sample.get("prompt", ""))
        image_path = sample.get("image_path", sample.get("image", None))
        ground_truth = sample.get("ground_truth", sample.get("answer", ""))
        data_source = sample.get("data_source", "unknown")
        sample_id = sample.get("id", sample.get("sample_id", str(sample_idx)))
        
        # Load image for multimodal generation (PIL object from the benchmark
        # loader takes precedence over a path on disk). Resolve it BEFORE the
        # prompt so the prompt can emit a matching vision placeholder.
        pil_image = sample.get("image_pil", None)
        if pil_image is None and image_path and isinstance(image_path, str) and os.path.exists(image_path):
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    pil_image = img.convert("RGB").copy()
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")

        # Build prompt (emit a vision placeholder iff an image will be fed in)
        prompt = self._build_prompt(question, image_path, has_image=pil_image is not None)

        # Generate response with the tool-integrated reasoning loop
        response = ""
        if self.inference_engine is not None:
            try:
                response = self._generate_with_tools(prompt, pil_image)
            except Exception as e:
                logger.error(f"Generation error for sample {sample_idx}: {e}")
                response = f"[ERROR: {str(e)}]"
        else:
            # Mock mode: honest placeholder — never echoes the ground truth.
            response = "[MOCK MODE: no inference engine available]"
        
        # Extract results
        prediction = self._extract_answer(response)
        confidence = self._extract_confidence(response)
        num_steps = self._count_steps(response)
        tool_calls = self._count_tool_calls(response)
        
        # Compute correctness
        is_correct = compute_exact_match(prediction, ground_truth)
        
        elapsed_time = time.time() - start_time
        
        result = {
            "sample_id": sample_id,
            "data_source": data_source,
            "question": question,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "confidence": confidence,
            "is_correct": is_correct,
            "num_steps": num_steps,
            "num_repairs": 0,  # TODO: Track repairs in full SERC
            "tool_calls": tool_calls,
            "tool_results": [],  # TODO: Execute tools
            "full_response": response,
            "elapsed_time": elapsed_time,
        }
        
        return result
    
    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Batch evaluation with metrics aggregation
        
        Args:
            dataset: List of sample dictionaries
            output_path: Path to save results
            resume_from: Path to resume from intermediate results
            
        Returns:
            Dictionary with metrics and per-sample results
        """
        logger.info(f"Starting evaluation on {len(dataset)} samples")
        
        # Handle resume
        completed_results = []
        start_idx = 0
        
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming from {resume_from}")
            with open(resume_from, "r") as f:
                checkpoint = json.load(f)
                completed_results = checkpoint.get("results", [])
                start_idx = len(completed_results)
                logger.info(f"Loaded {start_idx} completed samples")
        
        # Evaluate remaining samples
        results = completed_results.copy()
        
        pbar = tqdm(
            enumerate(dataset[start_idx:], start=start_idx),
            total=len(dataset),
            initial=start_idx,
            desc="Evaluating",
        )
        
        for idx, sample in pbar:
            try:
                result = self.evaluate_sample(sample, idx)
                results.append(result)
                
                # Update progress bar
                correct_so_far = sum(1 for r in results if r.get("is_correct", False))
                accuracy = correct_so_far / len(results) if results else 0
                pbar.set_postfix({"acc": f"{accuracy:.2%}", "correct": correct_so_far})
                
                # Save intermediate results
                if (
                    self.config.save_intermediate and 
                    (idx + 1) % self.config.intermediate_save_interval == 0
                ):
                    self._save_intermediate(results, output_path)
                    
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                # Record prediction/ground_truth so aggregate_metrics (which
                # recomputes correctness from these fields) scores the failed
                # sample as wrong instead of matching two empty strings.
                results.append({
                    "sample_id": str(sample.get("id", sample.get("sample_id", idx))),
                    "data_source": sample.get("data_source", "unknown"),
                    "ground_truth": sample.get("ground_truth", sample.get("answer", "")),
                    "prediction": "",
                    "error": str(e),
                    "is_correct": False,
                })
        
        # Aggregate metrics
        metrics = aggregate_metrics(results)
        
        # Prepare output
        output = {
            "config": self.config.to_dict(),
            "metrics": metrics.to_dict(),
            "results": results if self.config.save_per_sample_results else [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save results
        if output_path:
            self._save_results(output, output_path, metrics)
        
        # Print summary
        logger.info("\n" + format_metrics_table(metrics))
        
        return output
    
    def _save_intermediate(
        self,
        results: List[Dict[str, Any]],
        output_path: Optional[str],
    ):
        """Save intermediate results for resume capability"""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "intermediate.json")
        else:
            output_path = output_path.replace(".json", "_intermediate.json")
        
        checkpoint = {
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(output_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.debug(f"Saved intermediate results to {output_path}")
    
    def _save_results(
        self,
        output: Dict[str, Any],
        output_path: str,
        metrics: EvaluationMetrics,
    ):
        """Save final results in multiple formats"""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Saved results to {json_path}")
        
        # Save CSV metrics
        csv_path = json_path.replace(".json", "_metrics.csv")
        save_metrics_to_csv(metrics, csv_path)
        logger.info(f"Saved metrics to {csv_path}")
        
        # Save per-sample results as separate CSV
        if self.config.save_per_sample_results and output.get("results"):
            samples_csv_path = json_path.replace(".json", "_samples.csv")
            self._save_samples_csv(output["results"], samples_csv_path)
            logger.info(f"Saved per-sample results to {samples_csv_path}")
    
    def _save_samples_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save per-sample results to CSV"""
        import csv
        
        if not results:
            return
        
        # Get all keys
        keys = ["sample_id", "data_source", "is_correct", "prediction", "ground_truth", 
                "confidence", "num_steps", "num_repairs", "elapsed_time"]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    def evaluate_benchmarks(
        self,
        benchmark_names: List[str],
        data_dir: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate on multiple benchmarks
        
        Args:
            benchmark_names: List of benchmark names
            data_dir: Directory containing benchmark data
            output_dir: Output directory for results
            
        Returns:
            Dictionary mapping benchmark names to metrics
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = {}
        failed_benchmarks = {}

        for benchmark_name in benchmark_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating benchmark: {benchmark_name}")
            logger.info(f"{'='*60}")

            try:
                # Load benchmark
                benchmark = self._load_benchmark(benchmark_name, data_dir)

                if not benchmark:
                    raise RuntimeError(f"Benchmark '{benchmark_name}' loaded 0 samples")

                # Evaluate
                output_path = os.path.join(output_dir, f"{benchmark_name}_results.json")
                output = self.evaluate_dataset(benchmark, output_path)

                # Store metrics
                all_metrics[benchmark_name] = EvaluationMetrics(**output["metrics"])

            except Exception as e:
                logger.error(f"Error evaluating {benchmark_name}: {e}")
                failed_benchmarks[benchmark_name] = str(e)
                continue

        if failed_benchmarks:
            logger.error(f"FAILED benchmarks (no results produced): {failed_benchmarks}")
            if not all_metrics:
                raise RuntimeError(
                    f"All requested benchmarks failed to evaluate: {failed_benchmarks}. "
                    "Check that benchmark data exists (see verl/evaluation/benchmarks) "
                    "or that the HuggingFace datasets are reachable."
                )

        # Save combined results
        combined_path = os.path.join(output_dir, "combined_results.json")
        combined = {
            "benchmarks": {name: m.to_dict() for name, m in all_metrics.items()},
            "failed_benchmarks": failed_benchmarks,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        
        # Print summary table
        self._print_benchmark_summary(all_metrics)
        
        return all_metrics
    
    def _load_benchmark(
        self,
        benchmark_name: str,
        data_dir: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Load benchmark data via the registry in verl.evaluation.benchmarks.

        Raises a ValueError for unknown benchmark names and lets loader errors
        propagate — a missing benchmark must never be silently skipped.
        """
        from verl.evaluation.benchmarks import get_benchmark

        benchmark = get_benchmark(benchmark_name, data_dir)
        return benchmark.load_data()
    
    def _print_benchmark_summary(self, all_metrics: Dict[str, EvaluationMetrics]):
        """Print summary table of all benchmarks"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Benchmark':<20} {'Accuracy':>10} {'Fuzzy Acc':>10} {'Samples':>10}")
        print("-" * 80)
        
        for name, metrics in sorted(all_metrics.items()):
            print(
                f"{name:<20} "
                f"{metrics.accuracy * 100:>9.2f}% "
                f"{metrics.fuzzy_accuracy * 100:>9.2f}% "
                f"{metrics.total_samples:>10}"
            )
        
        print("=" * 80)
