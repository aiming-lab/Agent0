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
Verification parsing and scoring utilities for Agent0-VL
Handles parsing of step-by-step verification outputs and repair instructions
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any


def parse_verification_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse verification output from Agent0-VL Verifier

    Expected JSON format:
    {
      "step_index": 1,
      "score": 0.8,
      "confidence": 0.9,
      "critique": "Step is logically sound",
      "tool_check": false
    }

    Args:
        text: Verifier output text

    Returns:
        Parsed verification dict or None if parsing fails
    """
    # Try to extract JSON block
    json_pattern = r'\{[^{}]*"step_index"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        try:
            verification = json.loads(matches[-1])  # Take last match
            # Validate required fields
            required_fields = ['step_index', 'score', 'confidence', 'critique']
            if all(field in verification for field in required_fields):
                return verification
        except json.JSONDecodeError:
            pass

    return None


def parse_repair_instruction(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse repair instruction from Agent0-VL self-repair output

    Expected JSON format:
    {
      "action": "PATCH" | "NO_CHANGE",
      "target_step": 1,
      "patch_type": "text|code|tool_call|parameter",
      "new_content": "...",
      "justification": "..."
    }

    Args:
        text: Repair output text

    Returns:
        Parsed repair instruction dict or None
    """
    json_pattern = r'\{[^{}]*"action"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        try:
            repair = json.loads(matches[-1])
            if 'action' in repair and repair['action'] in ['PATCH', 'NO_CHANGE']:
                return repair
        except json.JSONDecodeError:
            pass

    return None


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract final answer from solver output

    Expected format:
    CONFIDENCE: 0.9
    FINAL_ANSWER: <answer>

    Args:
        text: Solver output text

    Returns:
        Extracted answer or None
    """
    # Look for FINAL_ANSWER marker
    pattern = r'FINAL_ANSWER:\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    if matches:
        return matches[-1].strip()

    return None


def extract_confidence(text: str) -> Optional[float]:
    """
    Extract confidence score from solver output

    Expected format:
    CONFIDENCE: 0.9

    Args:
        text: Solver output text

    Returns:
        Confidence score (0-1) or None
    """
    # Capture an optional leading sign so out-of-range values (e.g. "-0.5")
    # are parsed and then clamped, rather than silently failing to match.
    pattern = r'CONFIDENCE:\s*(-?[\d.]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        try:
            conf = float(matches[-1])
            return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except ValueError:
            pass

    return None


def parse_reasoning_steps(text: str) -> List[Dict[str, str]]:
    """
    Parse reasoning steps from solver output

    Extracts content within <think>...</think> tags

    Args:
        text: Solver output text

    Returns:
        List of reasoning step dictionaries
    """
    steps = []
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)

    for idx, content in enumerate(matches, 1):
        steps.append({
            'step_index': idx,
            'content': content.strip()
        })

    return steps


def compute_step_verification_score(
    verification: Dict[str, Any],
    tool_success: bool = True
) -> float:
    """
    Compute verification score for a single step

    Args:
        verification: Parsed verification output
        tool_success: Whether any tool calls succeeded

    Returns:
        Verification score in [-1, 1]
    """
    score = verification.get('score', 0.0)
    confidence = verification.get('confidence', 0.5)
    tool_check = verification.get('tool_check', False)

    # Combine score and confidence
    weighted_score = score * confidence

    # Apply tool success factor
    if tool_check and not tool_success:
        weighted_score *= 0.5  # Penalize if tool-based verification failed

    return max(-1.0, min(1.0, weighted_score))


def compute_process_reward(
    verifications: List[Dict[str, Any]],
    tool_rewards: List[float],
    lambda_tool: float = 0.3,
    beta_div: float = 0.01
) -> List[float]:
    """
    Compute process-level rewards for reasoning trajectory

    r_proc^(t) = λ_tool · r(tool_t) + score_t · conf_t - β_div · D_KL

    Args:
        verifications: List of verification outputs for each step
        tool_rewards: List of tool-based rewards for each step
        lambda_tool: Weight for tool-based correctness
        beta_div: Weight for cross-role regularization

    Returns:
        List of process rewards
    """
    process_rewards = []

    for i, verification in enumerate(verifications):
        score = verification.get('score', 0.0)
        conf = verification.get('confidence', 0.5)

        # Semantic reliability term
        semantic_reward = score * conf

        # Tool-based correctness term
        tool_reward = tool_rewards[i] if i < len(tool_rewards) else 0.0
        tool_term = lambda_tool * tool_reward

        # Process reward (KL term is approximated/omitted for simplicity)
        r_proc = tool_term + semantic_reward - beta_div * 0.1  # Placeholder KL

        process_rewards.append(r_proc)

    return process_rewards


def compute_repair_penalty(
    confidence: float,
    tau_c: float = 0.7,
    eta: float = 0.05,
    kappa: float = 5.0
) -> Tuple[float, float]:
    """
    Compute repair gate activation and penalty

    g_t = σ(κ(τ_c - conf_t))

    Args:
        confidence: Confidence score for the step
        tau_c: Repair threshold
        eta: Repair penalty coefficient
        kappa: Gate steepness parameter

    Returns:
        Tuple of (gate_activation, repair_cost)
    """
    import math

    # Sigmoid gate activation
    gate_input = kappa * (tau_c - confidence)
    gate_activation = 1.0 / (1.0 + math.exp(-gate_input))

    # Repair cost
    repair_cost = gate_activation * eta

    return gate_activation, repair_cost


def compute_total_return(
    process_rewards: List[float],
    outcome_reward: float,
    repair_penalties: List[float],
    alpha_out: float = 1.0,
    gamma: float = 0.99
) -> float:
    """
    Compute total trajectory return

    g(τ) = α_out · r_out + Σ γ^(t-1) (r_t - C_repair^(t))

    Args:
        process_rewards: List of process rewards
        outcome_reward: Final outcome reward
        repair_penalties: List of repair penalties
        alpha_out: Weight for outcome reward
        gamma: Discount factor

    Returns:
        Total return value
    """
    total_return = alpha_out * outcome_reward

    for t, (r_proc, r_penalty) in enumerate(zip(process_rewards, repair_penalties)):
        effective_reward = r_proc - r_penalty
        total_return += (gamma ** t) * effective_reward

    return total_return


def get_verification_accuracy(
    predicted_correct: bool,
    actual_correct: bool
) -> float:
    """
    Compute verification accuracy

    Args:
        predicted_correct: Whether verifier predicted correctness
        actual_correct: Ground truth correctness

    Returns:
        Binary accuracy (1.0 or 0.0)
    """
    return 1.0 if predicted_correct == actual_correct else 0.0


def parse_trajectory_with_tools(text: str) -> List[Dict[str, Any]]:
    """
    Parse complete trajectory including reasoning, tool calls, and verification

    Args:
        text: Complete trajectory text

    Returns:
        List of trajectory steps with all components
    """
    trajectory = []

    # Split by think tags
    think_pattern = r'<think>(.*?)</think>(.*?)(?=<think>|$)'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)

    for idx, (think_content, after_think) in enumerate(matches, 1):
        step = {
            'step_index': idx,
            'reasoning': think_content.strip(),
            'tool_calls': [],
            'tool_outputs': [],
            'verification': None
        }

        # Look for tool calls in after_think section
        tool_pattern = r'\{"tool_name":"([^"]+)","tool_input":(\{[^}]+\})\}'
        tool_matches = re.findall(tool_pattern, after_think)

        for tool_name, tool_input_str in tool_matches:
            try:
                tool_input = json.loads(tool_input_str)
                step['tool_calls'].append({
                    'tool_name': tool_name,
                    'tool_input': tool_input
                })
            except json.JSONDecodeError:
                pass

        trajectory.append(step)

    return trajectory


# Verification threshold constants
VERIFICATION_SCORE_THRESHOLD = 0.0  # Steps with score > 0 are considered correct
CONFIDENCE_THRESHOLD = 0.7  # Steps with confidence < 0.7 may need repair
REPAIR_THRESHOLD = 0.7  # Same as tau_c in paper
