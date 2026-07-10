# Copyright 2024 PRIME team and/or its affiliates
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

from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .multiturn import MultiTurnRewardManager
from .pag import PAGRewardManager
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .standalone_genrm import StandaloneGenRMRewardManager
from .agent0 import Agent0RewardManager

__all__ = [
    'NaiveRewardManager',
    'PrimeRewardManager',
    'MultiTurnRewardManager',
    'PAGRewardManager',
    'BatchRewardManager',
    'DAPORewardManager',
    'StandaloneGenRMRewardManager',
    'Agent0RewardManager'
]