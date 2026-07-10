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
from verl.utils.reward_score import compute_sympy_score,compute_deepscaler_score,compute_format_score,compute_reflection_pattern_score,compute_repeat_score
import torch
from concurrent.futures import ThreadPoolExecutor

class HackRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.name = compute_score
        self.compute_score_mapper = {
            "reflection" : compute_reflection_pattern_score,
            "repeat" : compute_repeat_score,
            'format' : compute_format_score,
            'math' : compute_sympy_score,
            'deepscaler' : compute_deepscaler_score,
        }
        if compute_score in self.compute_score_mapper:
            self.compute_score = self.compute_score_mapper[compute_score]
        else:
            self.compute_score = compute_sympy_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # Create tensors for all score types
        score_tensors = {
            'format_hack_scores': torch.zeros_like(data.batch['responses'], dtype=torch.float32),
            'reflection_hack_scores': torch.zeros_like(data.batch['responses'], dtype=torch.float32),
        }

        already_print_data_sources = {}

        def process_item(args):
            i, data_item = args
            
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            # only given response
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # Compute the main score
            main_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # Compute all other scores
            all_scores = {
                'format': self.compute_score_mapper['format'](
                    data_source=data_source, solution_str=response_str, 
                    ground_truth=ground_truth, extra_info=extra_info),
                'reflection': self.compute_score_mapper['reflection'](
                    data_source=data_source, solution_str=response_str, 
                    ground_truth=ground_truth, extra_info=extra_info),
            }
            
            # Return the data_source as well for printing logic
            return i, main_score, valid_response_length, sequences_str, data_source, all_scores

        # Process items in parallel using ThreadPoolExecutor
        args = [(i, data[i]) for i in range(len(data))]
        with ThreadPoolExecutor(max_workers=96) as executor:
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results and handle printing
        for i, score, valid_response_length, sequences_str, data_source, all_scores in results:
            reward_tensor[i, valid_response_length - 1] = score
            
            # Fill in all score tensors
            score_tensors['format_hack_scores'][i, valid_response_length - 1] = all_scores['format']
            score_tensors['reflection_hack_scores'][i, valid_response_length - 1] = all_scores['reflection']
            
            # Handle printing logic
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # Add all scores to the batch
        for score_name, score_tensor in score_tensors.items():
            data.batch[score_name] = score_tensor
        
        # Return the main reward tensor as the primary score
        return reward_tensor
