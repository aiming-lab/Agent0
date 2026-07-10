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

"""Prompt templates for Agent0-VL"""

from .agent0_templates import (
    SOLVER_SYSTEM_PROMPT,
    VERIFIER_PROMPT_TEMPLATE,
    REPAIR_PROMPT_TEMPLATE,
    get_solver_prompt,
    get_verifier_prompt,
    get_repair_prompt,
    validate_solver_output,
    validate_verifier_output,
    validate_repair_output
)

__all__ = [
    'SOLVER_SYSTEM_PROMPT',
    'VERIFIER_PROMPT_TEMPLATE',
    'REPAIR_PROMPT_TEMPLATE',
    'get_solver_prompt',
    'get_verifier_prompt',
    'get_repair_prompt',
    'validate_solver_output',
    'validate_verifier_output',
    'validate_repair_output'
]
