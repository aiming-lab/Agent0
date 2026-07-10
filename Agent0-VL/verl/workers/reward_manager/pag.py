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
from verl.utils.reward_score.genrm_verify import get_verification_score
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class PAGRewardManager:
    """Multi-turn dialogue reward manager with GenRM verification.
    
    Flow: User question -> Model answer -> Verification -> Regeneration (if needed)
    """

    def __init__(self, tokenizer, num_examine, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        config = config or {}
        self.max_turns = config.get('num_turns', 2)
        self.is_only_genrm = config.get('is_only_genrm', False)
        self.policy_rs = config.get('policy_rs', False)
        self.rs_coef = config.get('rs_coef', 10.0)
        self.end_with_verifer = config.get('end_with_verifer', False)

    def __call__(self, data: DataProto, return_dict: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute rewards for multi-turn dialogue."""
        if 'rm_scores' in data.batch:
            return data.batch['rm_scores'], {}

        batch_size = data.batch['responses'].shape[0]
        device = data.batch['responses'].device
        if 'num_turns' in data.meta_info:
            self.max_turns = data.meta_info['num_turns']
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metrics_tensors = {
            'turn_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
            'verify_accuracies': torch.zeros((batch_size, self.max_turns), dtype=torch.float32, device=device),
            'turn_counts': torch.zeros(batch_size, dtype=torch.long, device=device)
        }
        
        answer_logs = []
        reward_extra_info = defaultdict(list)
        printed_sources = {}
        
        for i in range(batch_size):
            # Initialize extra info for verifier mode
            if self.end_with_verifer:
                for key in ["all_pred", "all_acc", "all_genrm_pred", "all_genrm_score", "all_genrm_probs"]:
                    reward_extra_info[key].append([])

            data_item = data[i]
            response_ids = data_item.batch['responses']
            multiturn_mask = data_item.batch['multiturn_mask']
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            
            # Find turn boundaries from mask
            turn_boundaries = []
            if multiturn_mask.numel() > 0:
                padded_mask = torch.cat([torch.tensor([True], device=multiturn_mask.device), multiturn_mask])
                diff = padded_mask[1:].long() - padded_mask[:-1].long()
                turn_boundaries = torch.where(diff == -1)[0].tolist()
                if multiturn_mask[-1]:
                    turn_boundaries.append(multiturn_mask.size(0))
            
            metrics_tensors['turn_counts'][i] = len(turn_boundaries) - 1
            sample_answers = []
                
            # Process first turn
            first_response = self.tokenizer.decode(response_ids[:turn_boundaries[0]])
            first_result = get_policy_score(solution_str=first_response, ground_truth=ground_truth)
            
            reward_extra_info["pred"].append(first_result["pred"])
            reward_extra_info["acc"].append(first_result["acc"])
            reward_extra_info["length"].append(turn_boundaries[0])
            if self.end_with_verifer:
                reward_extra_info["all_pred"][-1].append(first_result["pred"])
                reward_extra_info["all_acc"][-1].append(first_result["acc"])
            
            reward_tensor[i, turn_boundaries[0] - 1] = first_result["acc"]
            metrics_tensors['turn_accuracies'][i, 0] = first_result["acc"] >= 0.5
            sample_answers.append(first_result['pred'])
            
            gt_judge = first_result["acc"] >= 0.5
            prev_acc = first_result["acc"]
            max_turns = self.max_turns + 1 if self.end_with_verifer else self.max_turns
                
            # Process subsequent turns
            for turn in range(1, max_turns):
                # GenRM verification
                verify_start = turn_boundaries[2*turn - 2] if turn > 1 else turn_boundaries[0]
                verify_end = turn_boundaries[2*turn - 1]
                verify_response = self.tokenizer.decode(
                    response_ids[verify_start:verify_end][multiturn_mask[verify_start:verify_end]], 
                    skip_special_tokens=True
                )
                
                verify_result = get_verification_score(verify_response, gt_judge)
                reward_tensor[i, verify_end - 1] = verify_result["genrm_score"]
                metrics_tensors['verify_accuracies'][i, turn-1] = verify_result["genrm_score"]
                
                # Store verification info
                if turn == 1:
                    reward_extra_info["genrm_pred"].append(verify_result["genrm_pred"])
                    reward_extra_info["genrm_score"].append(verify_result["genrm_score"])
                    reward_extra_info["genrm_probs"].append(data_item.non_tensor_batch["verify_probs"])
                if self.end_with_verifer:
                    reward_extra_info["all_genrm_pred"][-1].append(verify_result["genrm_pred"])
                    reward_extra_info["all_genrm_score"][-1].append(verify_result["genrm_score"])
                    reward_extra_info['all_genrm_probs'][-1].append(data_item.non_tensor_batch["verify_probs"])
                
                # Policy response (if exists)
                if 2*turn >= len(turn_boundaries) or (self.end_with_verifer and turn == max_turns - 1):
                    break
                
                if self.is_only_genrm:
                    multiturn_mask[:verify_end] = False
                
                policy_start = verify_end
                policy_end = turn_boundaries[2*turn]
                policy_response = self.tokenizer.decode(
                    response_ids[policy_start:policy_end][multiturn_mask[policy_start:policy_end]]
                )
                
                policy_result = get_policy_score(solution_str=policy_response, ground_truth=ground_truth)
                
                # Set reward with optional reward shaping
                reward_value = policy_result["acc"]
                if self.policy_rs:
                    reward_value += self.rs_coef * (policy_result["acc"] - prev_acc)
                reward_tensor[i, policy_end - 1] = reward_value
                
                metrics_tensors['turn_accuracies'][i, turn] = policy_result["acc"]
                prev_acc = policy_result["acc"]
                gt_judge = policy_result["acc"] >= 0.5
                
                if turn == 1:
                    sample_answers.append(policy_result['pred'])
                    answer_logs.append(sample_answers)
                if self.end_with_verifer:
                    reward_extra_info["all_pred"][-1].append(policy_result["pred"])
                    reward_extra_info["all_acc"][-1].append(policy_result["acc"])
            
            # Debug output
            if self.num_examine > 0 and printed_sources.get(data_source, 0) < self.num_examine:
                printed_sources[data_source] = printed_sources.get(data_source, 0) + 1
                full_sequence = torch.cat((data_item.batch['prompts'], response_ids))
                print(self.tokenizer.decode(full_sequence, skip_special_tokens=True))
            if self.end_with_verifer:
                reward_extra_info["response"].append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        
        # Compute metrics
        data_sources = None
        if data.meta_info.get('validate', False):
            data_sources = [data[i].non_tensor_batch.get('data_source', 'unknown') for i in range(len(data))]
        
        metrics = self._compute_metrics(metrics_tensors, data_sources, answer_logs, data.non_tensor_batch["final_generation_turn"])
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info, "metrics": metrics}
        return reward_tensor, metrics 
        
    def _compute_single_group_metrics(self, turn_acc, verify_acc, turn_counts, final_turn, prefix=""):
        """Compute metrics for a group (global or source-specific)."""
        metrics = {}
        
        # Basic accuracy
        final_acc = turn_acc.gather(dim=-1, index=final_turn.unsqueeze(-1))
        metrics[f'{prefix}final_acc'] = final_acc.mean().item()
        
        for i in range(self.max_turns):
            clamped_turn = final_turn.clone().clamp(max=i)
            turn_policy_acc = turn_acc.gather(dim=-1, index=clamped_turn.unsqueeze(-1))
            metrics[f'{prefix}turn_{i+1}_accuracy'] = turn_policy_acc.mean().item()
        
        # Turn distribution
        for i in range(2*self.max_turns-1):
            count = (turn_counts == i).sum().item()
            if count > 0:
                metrics[f'{prefix}turn_count_{i}'] = count
                metrics[f'{prefix}turn_count_{i}_ratio'] = count / len(turn_counts)
        
        # Turn-specific accuracy
        for i in range(1, self.max_turns):
            policy_mask = turn_counts >= i*2
            if policy_mask.any():
                metrics[f'{prefix}turn_{i+1}_accuracy_selection'] = turn_acc[policy_mask, i].mean().item()
            
            verify_mask = turn_counts >= i*2-1
            if verify_mask.any():
                metrics[f'{prefix}verify_{i}_accuracy'] = verify_acc[:, i-1].mean().item()
        
        # Confusion matrix for verification
        if len(turn_acc) > 0:
            turn1_policy = turn_acc[:, 0]
            turn1_verify = verify_acc[:, 0]
            
            TP = ((turn1_verify > 0.5) & (turn1_policy > 0.5)).sum().item()
            FP = ((turn1_verify <= 0.5) & (turn1_policy <= 0.5)).sum().item()
            FN = ((turn1_verify <= 0.5) & (turn1_policy > 0.5)).sum().item()
            TN = ((turn1_verify > 0.5) & (turn1_policy <= 0.5)).sum().item()
            
            metrics[f'{prefix}verify_TP'] = TP
            metrics[f'{prefix}verify_FP'] = FP
            metrics[f'{prefix}verify_FN'] = FN
            metrics[f'{prefix}verify_TN'] = TN
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                f'{prefix}verify_precision': precision,
                f'{prefix}verify_recall': recall,
                f'{prefix}verify_recall_negative': recall_neg,
                f'{prefix}verify_f1': f1
            })
            
            # Transition metrics
            if turn_acc.shape[1] > 1:
                turn2_mask = (turn_counts == 2)
                if turn2_mask.any():
                    turn2_policy = turn_acc[:, 1][turn2_mask]
                    turn1_policy_masked = turn1_policy[turn2_mask]
                    
                    i_to_c = ((turn2_policy > 0.5) & (turn1_policy_masked <= 0.5)).sum().item()
                    c_to_i = ((turn2_policy <= 0.5) & (turn1_policy_masked > 0.5)).sum().item()
                    
                    total_incorrect = (turn1_policy_masked <= 0.5).sum().item()
                    total_correct = (turn1_policy_masked > 0.5).sum().item()
                    
                    if total_incorrect > 0:
                        metrics.update({
                            f'{prefix}i_to_c_rate': i_to_c / total_incorrect,
                            f'{prefix}i_to_c_rate_gt': i_to_c / len(turn1_policy),
                            f'{prefix}i_to_c_count': i_to_c
                        })
                    
                    if total_correct > 0:
                        metrics.update({
                            f'{prefix}c_to_i_rate': c_to_i / total_correct,
                            f'{prefix}c_to_i_rate_gt': c_to_i / len(turn1_policy),
                            f'{prefix}c_to_i_count': c_to_i
                        })
        
        return metrics

    def _compute_metrics(self, metrics_tensors, data_sources=None, answer_logs=None, final_generation_turn=None):
        """Compute all metrics."""
        final_turn = torch.tensor(final_generation_turn, device=metrics_tensors['turn_accuracies'].device, dtype=torch.long)
        print("final_answer_turn", final_turn)
        
        # Global metrics
        metrics = self._compute_single_group_metrics(
            metrics_tensors['turn_accuracies'], metrics_tensors['verify_accuracies'], 
            metrics_tensors['turn_counts'], final_turn
        )
        
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
            data_sources = np.array(data_sources) if isinstance(data_sources, list) else data_sources
            for source in np.unique(data_sources):
                indices = torch.tensor(np.where(data_sources == source)[0], device=final_turn.device)
                if len(indices) > 0:
                    source_metrics = self._compute_single_group_metrics(
                        metrics_tensors['turn_accuracies'][indices],
                        metrics_tensors['verify_accuracies'][indices],
                        metrics_tensors['turn_counts'][indices],
                        final_turn[indices],
                        prefix=f'{source}/'
                    )
                    metrics.update(source_metrics)
        
        return metrics 