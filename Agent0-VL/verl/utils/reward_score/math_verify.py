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

from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify.errors import TimeoutException


def compute_score(solution_str: str, ground_truth: str) -> dict:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    )
    ret_score = 0.
    pred = ""

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters
    try:
        ret_score, answers = verify_func([ground_truth_boxed], [solution_str])
        if answers and len(answers) > 1 and len(answers[1]) > 0:
            pred = answers[1][0]
    except TimeoutException as e:
        pass
    except Exception as e:
        pass
        # print(e)

    return {
        "score": ret_score,
        "acc": ret_score,
        "pred": pred,
    }
