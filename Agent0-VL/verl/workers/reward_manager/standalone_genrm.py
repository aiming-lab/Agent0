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
from verl.utils.reward_score.genrm_verify import get_verification_score
import torch
from collections import defaultdict


class StandaloneGenRMRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        policy_acc = []
        verify_acc = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch["data_source"]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            acc = data_item.non_tensor_batch["reward_model"]["acc"]
            policy_acc.append(acc)

            score = get_verification_score(
                solution_str = response_str,
                gt_judge = acc > 0.5,
            )

            if isinstance(score, dict):
                reward = score["genrm_score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                
                if "genrm_score" in score:
                    verify_acc.append(score["genrm_score"])
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
        
        metrics = {}
        if policy_acc and verify_acc:
            policy_acc_tensor = torch.tensor(policy_acc)
            verify_acc_tensor = torch.tensor(verify_acc)
            
            TP = torch.sum((verify_acc_tensor > 0.5) & (policy_acc_tensor > 0.5)).item()
            FP = torch.sum((verify_acc_tensor <= 0.5) & (policy_acc_tensor <= 0.5)).item()
            FN = torch.sum((verify_acc_tensor <= 0.5) & (policy_acc_tensor > 0.5)).item()
            TN = torch.sum((verify_acc_tensor > 0.5) & (policy_acc_tensor <= 0.5)).item()
            
            metrics['verify_TP'] = TP
            metrics['verify_FP'] = FP
            metrics['verify_FN'] = FN
            metrics['verify_TN'] = TN
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0 
            recall_negative = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics['verify_precision'] = precision
            metrics['verify_recall'] = recall
            metrics['verify_recall_negative'] = recall_negative
            metrics['verify_f1'] = f1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "metrics": metrics
            }
        else:
            return reward_tensor
