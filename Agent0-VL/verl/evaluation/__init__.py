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
Agent0-VL Evaluation Module
Comprehensive evaluation pipeline for vision-language benchmarks
"""

from verl.evaluation.metrics import (
    compute_accuracy,
    compute_fuzzy_accuracy,
    compute_verification_f1,
    compute_repair_rate,
    compute_tool_success_rate,
    aggregate_metrics,
    EvaluationMetrics,
)
from verl.evaluation.agent0_evaluator import Agent0Evaluator

__all__ = [
    "Agent0Evaluator",
    "compute_accuracy",
    "compute_fuzzy_accuracy",
    "compute_verification_f1",
    "compute_repair_rate",
    "compute_tool_success_rate",
    "aggregate_metrics",
    "EvaluationMetrics",
]
