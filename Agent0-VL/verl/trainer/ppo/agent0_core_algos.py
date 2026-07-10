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
Agent0-VL GRPO-specific advantage computation.
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import torch


def compute_agent0_grpo_advantage(
    process_rewards: torch.Tensor,
    outcome_reward: torch.Tensor,
    repair_penalties: torch.Tensor,
    response_mask: torch.Tensor,
    step_end_positions: torch.Tensor,
    index: np.ndarray,
    alpha_out: float = 1.0,
    gamma: float = 0.99,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO advantage with process rewards.

    Total return:
        g(τ) = α_out · r_out + Σ γ^(t-1) (r_proc^(t) - C_repair^(t))

    Returns are normalized within each prompt group.

    Args:
        process_rewards: `(torch.Tensor)`
            shape: (bs, max_steps)
        outcome_reward: `(torch.Tensor)`
            shape: (bs,)
        repair_penalties: `(torch.Tensor)`
            shape: (bs, max_steps)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        step_end_positions: `(torch.Tensor)`
            shape: (bs, max_steps), token positions (1-based) where each step ends
        index: `(np.ndarray)`
            prompt indices for grouping
        alpha_out: `(float)`
            outcome reward weight
        gamma: `(float)`
            discount factor
        epsilon: `(float)`
            numerical stability constant

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, max_steps = process_rewards.shape
        response_length = response_mask.shape[-1]
        device = process_rewards.device

        step_positions = step_end_positions.to(device=device, dtype=torch.long)
        step_mask = (step_positions > 0) & (step_positions <= response_length)
        step_mask = step_mask.to(process_rewards.dtype)

        discounts = torch.pow(
            torch.tensor(gamma, device=device, dtype=process_rewards.dtype),
            torch.arange(max_steps, device=device, dtype=process_rewards.dtype),
        )
        discounted_steps = (process_rewards - repair_penalties) * discounts
        discounted_steps = discounted_steps * step_mask

        total_returns = alpha_out * outcome_reward + discounted_steps.sum(dim=-1)

        id2scores = defaultdict(list)
        id2mean = {}
        id2std = {}

        for i in range(batch_size):
            id2scores[index[i]].append(total_returns[i])

        for idx, scores_list in id2scores.items():
            scores = torch.stack(scores_list).to(device=device, dtype=total_returns.dtype)
            if scores.numel() == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                mean = scores.mean()
                std = scores.std(unbiased=False)
                if std < epsilon:
                    std = torch.tensor(1.0, device=device)
                id2mean[idx] = mean
                id2std[idx] = std

        normalized_returns = total_returns.clone()
        for i in range(batch_size):
            normalized_returns[i] = (total_returns[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        advantages = normalized_returns.unsqueeze(-1) * response_mask

    return advantages, total_returns
