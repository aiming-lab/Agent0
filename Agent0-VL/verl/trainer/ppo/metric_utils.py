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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from functools import partial


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True, max_singleturn_resp_length: int = None) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    multiturn_mask = batch.batch['multiturn_mask'].bool()
    response_mask = response_mask & multiturn_mask

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # 找到每一轮对话的开始位置
    turn_starts = multiturn_mask & (~torch.roll(multiturn_mask, shifts=1, dims=1))
    turn_starts[:, 0] = multiturn_mask[:, 0]  # 第一个位置特殊处理
    
    # 计算每个token属于哪一轮
    turn_indices = torch.cumsum(turn_starts.long(), dim=1)
    
    # 动态推断对话最大轮次
    max_turns = turn_indices.max().item()
    
    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    for turn in range(1, max_turns + 1):
        turn_mask = (turn_indices == turn) & multiturn_mask
        turn_lengths = turn_mask.sum(dim=1).float()  # (batch_size,)
        pos_indices = torch.arange(multiturn_mask.size(1), device=multiturn_mask.device).unsqueeze(0).expand_as(multiturn_mask)
        last_pos = torch.max(torch.where(turn_mask, pos_indices, torch.zeros_like(pos_indices)), dim=1)[0]
        eos_value_indices = torch.clamp(last_pos + 1, 0, values.size(1) - 1)
        turn_values = torch.masked_select(values, turn_mask)
        turn_eos_value = torch.gather(values, dim=1, index=eos_value_indices.unsqueeze(1))

        valid_mask = turn_lengths > 0
        turn_lengths = turn_lengths[valid_mask]
        turn_eos_value = turn_eos_value[valid_mask]
        
        # 只计算至少有一个样本存在此轮对话的情况
        if turn_lengths.sum() > 0:
            turn_metrics = {
                f'response_length_turn{turn}/mean': torch.mean(turn_lengths).detach().item(),
                f'response_length_turn{turn}/max': torch.max(turn_lengths).detach().item(),
                f'response_length_turn{turn}/min': torch.min(turn_lengths[turn_lengths > 0]).detach().item(),
                
                f'response_length_turn{turn}/samples': (turn_lengths > 0).sum().item(),  # 有多少样本包含此轮对话
                f'response_length_turn{turn}/clip_ratio': torch.mean(
                    torch.eq(turn_lengths, max_singleturn_resp_length).float()
                ).detach().item(),
            }
            values_metrics = {
                f'critic/turn{turn}_eos_value/mean': torch.mean(turn_eos_value).detach().item(),
                f'critic/turn{turn}_eos_value/max': torch.max(turn_eos_value).detach().item(),
                f'critic/turn{turn}_eos_value/min': torch.min(turn_eos_value).detach().item(),
                f'critic/turn{turn}_values/min': torch.min(turn_values).detach().item(),
                f'critic/turn{turn}_values/max': torch.max(turn_values).detach().item(),
                f'critic/turn{turn}_values/mean': torch.mean(turn_values).detach().item(),
            }
            
            metrics.update(turn_metrics)
            metrics.update(values_metrics)
    
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(data: list[Any],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

def calc_maj_all_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    assert len(data[0][vote_key]) == len(data[0][val_key])
    for d in data:
        for i in range(len(d[vote_key])):
            vote2vals[d[vote_key][i]].append(d[val_key][i])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

def calc_maj_final_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key][-1]].append(d[val_key][-1])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def calc_genrm_val(data: list[dict[str, Any]], val_key: str, pred_key: str) -> float:
    correct_samples = [d for d in data if d[pred_key] == "correct"]
    if not correct_samples:
        return 0
    return np.mean([d[val_key] for d in correct_samples])


def calc_genrm_bo1_val(data: list[dict[str, Any]], val_key: str, pred_key: str, prob_key: str) -> float:
    correct_samples = [d for d in data if d[pred_key] == "correct" and d[prob_key] is not None]
    wrong_samples = [d for d in data if d[pred_key] == "wrong" and d[prob_key] is not None]
    if correct_samples:
        bo1_index = np.argmax([d[prob_key] for d in correct_samples])
        return correct_samples[bo1_index][val_key]
    elif wrong_samples:
        bo1_index = np.argmin([d[prob_key] for d in wrong_samples])
        return wrong_samples[bo1_index][val_key]
    else:
        return np.mean([d[val_key] for d in data])
    
def calc_genrm_all_bo1_val(data: list[dict[str, Any]], val_key: str, pred_key: str, prob_key: str) -> float:
    total = []
    for d in data:
        assert len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        for i in range(len(d[pred_key])):
            total.append((d[pred_key][i], d[prob_key][i], d[val_key][i]))
    correct_samples = [d for d in total if d[0] == "correct" and d[1] is not None]
    wrong_samples = [d for d in total if d[0] == "wrong" and d[1] is not None]
    if correct_samples:
        bo1_index = np.argmax([d[1] for d in correct_samples])
        return correct_samples[bo1_index][2]
    elif wrong_samples:
        bo1_index = np.argmin([d[1] for d in wrong_samples])
        return wrong_samples[bo1_index][2]
    else:
        return np.mean([d[2] for d in data])


def calc_genrm_final_bo1_val(data: list[dict[str, Any]], val_key: str, pred_key: str, prob_key: str) -> float:
    total = []
    for d in data:
        assert len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        total.append((d[pred_key][-1], d[prob_key][-1], d[val_key][-1]))
    correct_samples = [d for d in total if d[0] == "correct" and d[1] is not None]
    wrong_samples = [d for d in total if d[0] == "wrong" and d[1] is not None]
    if correct_samples:
        bo1_index = np.argmax([d[1] for d in correct_samples])
        return correct_samples[bo1_index][2]
    elif wrong_samples:
        bo1_index = np.argmin([d[1] for d in wrong_samples])
        return wrong_samples[bo1_index][2]
    else:
        return np.mean([d[2] for d in data])


def calc_genrm_weighted_val(data: list[dict[str, Any]], val_key: str, vote_key: str, pred_key: str, prob_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    vote2rm = defaultdict(int)

    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])
        if d[pred_key] == "correct":
            vote2rm[d[vote_key]] += d[prob_key]
        elif d[pred_key] == "wrong":
            vote2rm[d[vote_key]] += - d[prob_key]

    max_vote = max(vote2rm, key=vote2rm.get)
    maj_val = vote2vals[max_vote][0]
    return maj_val


def calc_genrm_weighted_all_val(data: list[dict[str, Any]], val_key: str, vote_key: str, pred_key: str, prob_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    vote2rm = defaultdict(int)

    for d in data:
        assert len(d[vote_key]) == len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[vote_key])={len(d[vote_key])}, len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        for i in range(len(d[vote_key])):
            vote2vals[d[vote_key][i]].append(d[val_key][i])
            if d[pred_key][i] == "correct":
                vote2rm[d[vote_key][i]] += d[prob_key][i]
            elif d[pred_key][i] == "wrong":
                vote2rm[d[vote_key][i]] += - d[prob_key][i]
    max_vote = max(vote2rm, key=vote2rm.get)
    maj_val = vote2vals[max_vote][0]
    return maj_val


def calc_genrm_weighted_final_val(data: list[dict[str, Any]], val_key: str, vote_key: str, pred_key: str, prob_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    # 保存data 
    np.save("data_debug/calc_genrm_all_data.npy", data)

    vote2vals = defaultdict(list)
    vote2rm = defaultdict(int)
    for d in data:
        assert len(d[vote_key]) == len(d[pred_key]) == len(d[prob_key]) == len(d[val_key]), \
            f"len(d[vote_key])={len(d[vote_key])}, len(d[pred_key])={len(d[pred_key])}, len(d[prob_key])={len(d[prob_key])}, len(d[val_key])={len(d[val_key])}"
        vote2vals[d[vote_key][-1]].append(d[val_key][-1])
        if d[pred_key][-1] == "correct":
            vote2rm[d[vote_key][-1]] += d[prob_key][-1]
        elif d[pred_key][-1] == "wrong":
            vote2rm[d[vote_key][-1]] += - d[prob_key][-1]

    try:
        max_vote = max(vote2rm, key=vote2rm.get)
        maj_val = vote2vals[max_vote][0]
    except:
        breakpoint()
    return maj_val


def process_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        seed: Random seed for bootstrapping
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        
        for var_name, var_vals in infos_dict.items():
            if sample_idx < len(var_vals):
                var2vals[var_name].append(var_vals[sample_idx])
    
    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if var_name not in ["acc", "all_acc"]:
                    continue
                metric = {}
                n_resps = len(var_vals)
                if var_name == "acc":
                    metric[f"mean@{n_resps}"] = np.mean(var_vals)
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    if var_name == "acc":
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                                                                                    subset_size=n,
                                                                                    reduce_fns=[np.max, np.min],
                                                                                    seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None and var_name == "acc":
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std
                    
                    if var2vals.get("all_pred", None) is not None and var_name == "all_acc":
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["all_pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_final_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj_final@{n}/mean"], metric[f"maj_final@{n}/std"] = maj_n_mean, maj_n_std
                    
                    if var2vals.get("all_pred", None) is not None and var_name == "all_acc":
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["all_pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_all_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj_all@{n}/mean"], metric[f"maj_all@{n}/std"] = maj_n_mean, maj_n_std

                    if var2vals.get("genrm_probs", None) is not None and var_name == "acc":
                        genrm_data = [{"val": val, "pred": pred, "prob": prob} for val, pred, prob in zip(var_vals, var2vals["genrm_pred"], var2vals["genrm_probs"])]
                        [(genrm_n_mean, genrm_n_std)
                        ] = bootstrap_metric(data=genrm_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_genrm_bo1_val, val_key="val", pred_key="pred", prob_key="prob")],
                                             seed=seed)
                        metric[f"genrm_prob_bo1@{n}/mean"], metric[f"genrm_prob_bo1@{n}/std"] = genrm_n_mean, genrm_n_std

                    if var2vals.get("genrm_probs", None) is not None and var_name == "all_acc":
                        genrm_data = [{"val": val, "pred": pred, "prob": prob} for val, pred, prob in zip(var_vals, var2vals["all_genrm_pred"], var2vals["all_genrm_probs"])]
                        [(genrm_n_mean, genrm_n_std)
                        ] = bootstrap_metric(data=genrm_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_genrm_final_bo1_val, val_key="val", pred_key="pred", prob_key="prob")],
                                             seed=seed)
                        metric[f"genrm_prob_final_bo1@{n}/mean"], metric[f"genrm_prob_final_bo1@{n}/std"] = genrm_n_mean, genrm_n_std
                    
                    if var2vals.get("genrm_probs", None) is not None and var_name == "all_acc":
                        genrm_data = [{"val": val, "pred": pred, "prob": prob} for val, pred, prob in zip(var_vals, var2vals["all_genrm_pred"], var2vals["all_genrm_probs"])]
                        [(genrm_n_mean, genrm_n_std)
                        ] = bootstrap_metric(data=genrm_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_genrm_all_bo1_val, val_key="val", pred_key="pred", prob_key="prob")],
                                             seed=seed)
                        metric[f"genrm_prob_all_bo1@{n}/mean"], metric[f"genrm_prob_all_bo1@{n}/std"] = genrm_n_mean, genrm_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
