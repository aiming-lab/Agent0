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

from verl import DataProto
from verl.utils.reward_score.math_verify import compute_score as get_policy_score
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class MultiTurnRewardManager:
    """Reward manager for multi-turn dialogue with optimized performance."""

    def __init__(self, tokenizer, num_examine, compute_score=None, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        config = config or {}
        self.max_turns = config.get('num_turns', 2)
        self.is_golden_generation = config.get('is_golden_generation', False)
        self.policy_rs = config.get('policy_rs', False)
        self.rs_coef = config.get('rs_coef', 10.0)

    def _get_turn_boundaries(self, multiturn_mask):
        """Extract turn boundaries from multiturn mask."""
        if multiturn_mask.numel() == 0:
            return []
        
        padded_mask = torch.cat([torch.tensor([True], device=multiturn_mask.device), multiturn_mask])
        diff = padded_mask[1:].long() - padded_mask[:-1].long()
        end_indices = torch.where(diff == -1)[0].tolist()
        
        if multiturn_mask[-1]:
            end_indices.append(multiturn_mask.size(0))
        
        return end_indices

    def _compute_single_group_metrics(self, turn_accuracies, prefix=""):
        """Compute metrics for a single group (global or source-specific).
        
        Args:
            turn_accuracies: Tensor of shape (batch_size, max_turns) containing accuracy scores
            max_turns: Maximum number of turns
            prefix: Prefix for metric names (e.g., "math/" for source-specific metrics)
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        metrics = {}
        
        # Turn-specific accuracy metrics
        for t in range(self.max_turns):
            turn_acc = turn_accuracies[:, t]
            metrics[f'{prefix}turn_{t+1}_accuracy'] = torch.mean(turn_acc).item()
        
        # Turn transition metrics
        for t in range(1, self.max_turns):
            change = turn_accuracies[:, t] - turn_accuracies[:, t-1]
            metrics[f'{prefix}turn_{t}_to_{t+1}_change'] = torch.mean(change).item()
            metrics[f'{prefix}turn_{t}_to_{t+1}_c2i'] = torch.sum(change < 0).item() / len(change)
            metrics[f'{prefix}turn_{t}_to_{t+1}_i2c'] = torch.sum(change > 0).item() / len(change)
        
        return metrics

    def _compute_metrics(self, metrics_tensors: Dict[str, torch.Tensor], data_sources=None, answer_logs=None) -> Dict[str, float]:
        """Compute all metrics, optionally grouped by data source.
        
        Args:
            metrics_tensors: Dictionary containing metric tensors
            max_turns: Maximum number of turns
            data_sources: List of data source information for samples, if provided then compute source-grouped metrics
            answer_logs: List of answer logs for analyzing answer changes per turn
            
        Returns:
            Dict[str, float]: Dictionary containing global metrics and source-grouped metrics
        """
        # Global metrics
        metrics = self._compute_single_group_metrics(metrics_tensors['turn_accuracies'])
        
        # Answer change analysis
        if answer_logs:
            regen_samples = answer_changed = 0
            for answers in answer_logs:
                if len(answers) >= 2 and all(a is not None for a in answers[:2]):
                    regen_samples += 1
                    if answers[0] != answers[1]:
                        answer_changed += 1
            
            if regen_samples > 0:
                metrics.update({
                    'answer_change_ratio': answer_changed / regen_samples,
                    'answer_changed_samples': answer_changed,
                    'regeneration_samples': regen_samples
                })
            
        # Source-specific metrics
        if data_sources is not None:
            if isinstance(data_sources, list):
                data_sources = np.array(data_sources)
            
            unique_sources = np.unique(data_sources)
            for source in unique_sources:
                source_indices = np.where(data_sources == source)[0]
                
                # Only process sources with samples
                if len(source_indices) == 0:
                    continue
                    
                source_indices_tensor = torch.tensor(source_indices, device=metrics_tensors['turn_accuracies'].device)
                source_metrics = self._compute_single_group_metrics(
                    metrics_tensors['turn_accuracies'][source_indices_tensor], 
                    prefix=f'{source}/'
                )
                metrics.update(source_metrics)
        
        return metrics

    def __call__(self, data: DataProto, return_dict: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute optimized rewards for multi-turn dialogue.
        
        Args:
            data: Input data proto containing batch and metadata
            return_dict: Whether to return results as dictionary
            
        Returns:
            Tuple[Tensor, Dict[str, Any]]: Reward tensor and dictionary containing various metrics
        """
        if 'rm_scores' in data.batch:
            return data.batch['rm_scores'], {}

        batch_size = data.batch['responses'].shape[0]
        device = data.batch['responses'].device
        if 'num_turns' in data.meta_info:
            self.max_turns = data.meta_info['num_turns']
        printed_sources = {}
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metrics_tensors = {
            'turn_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
        }
        
        answer_logs = [[] for _ in range(batch_size)]
        scores = [[] for _ in range(batch_size)]
        reward_extra_info = defaultdict(list)
        
        # Process each batch item
        for i in range(len(data)):
            for key in ["all_pred", "all_acc"]:
                reward_extra_info[key].append([])

            data_item = data[i]
            response_ids = data_item.batch['responses']
            multiturn_mask = data_item.batch['multiturn_mask']
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

            # Find turn boundaries from mask
            turn_boundaries = self._get_turn_boundaries(multiturn_mask)

            # Compute scores for each turn
            start_pos = 0
            for t, end_pos in enumerate(turn_boundaries):
                current_response = self.tokenizer.decode(response_ids[start_pos:end_pos])

                # Compute all scores in one pass
                result = get_policy_score(solution_str=current_response, ground_truth=ground_truth)
                answer_logs[i].append(result['pred'])
                scores[i].append(result['score'])
                metrics_tensors['turn_accuracies'][i, t] = result['acc']

                if self.policy_rs and t >= 1:
                    reward_tensor[i, end_pos - 1] = result['score'] + self.rs_coef * (result['score'] - scores[i][t-1])
                else:
                    reward_tensor[i, end_pos - 1] = result['score']

                if t == 0:
                    reward_extra_info["pred"].append(result["pred"])
                    reward_extra_info["acc"].append(result["acc"])
                reward_extra_info["all_pred"][-1].append(result['pred'])
                reward_extra_info["all_acc"][-1].append(result['acc'])

                start_pos = end_pos
            
            # Debug output
            if self.num_examine > 0 and printed_sources.get(data_source, 0) < self.num_examine:
                printed_sources[data_source] = printed_sources.get(data_source, 0) + 1
                full_sequence = torch.cat((data_item.batch['prompts'], response_ids))
                print(self.tokenizer.decode(full_sequence, skip_special_tokens=True))
        
        # Compute metrics
        data_sources = None
        if data.meta_info.get('validate', False):
            data_sources = [data[i].non_tensor_batch.get('data_source', 'unknown') for i in range(len(data))]
        
        # Compute metrics
        metrics = self._compute_metrics(metrics_tensors, data_sources, answer_logs)
        
        # Golden generation logic
        if self.is_golden_generation:
            for i in range(batch_size):
                boundaries = self._get_turn_boundaries(data.batch['multiturn_mask'][i])
                for t, end_pos in enumerate(boundaries):
                    if t < self.max_turns and metrics_tensors['turn_accuracies'][i, t] >= 0.5:
                        # If current turn is correct, set all positions from next turn onwards to False
                        if t + 1 < len(boundaries):
                            start_pos = boundaries[t]  # End position of current turn
                            data.batch['multiturn_mask'][i, start_pos + 1:] = False
                        break
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "metrics": metrics
            }
        else:
            return reward_tensor, metrics