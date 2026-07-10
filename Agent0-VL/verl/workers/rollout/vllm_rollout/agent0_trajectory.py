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
Agent0-VL Trajectory Management
Dataclasses and utilities for tracking multi-turn reasoning trajectories
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class StepStatus(Enum):
    """Status of a reasoning step"""
    PENDING = "pending"
    COMPLETED = "completed"
    VERIFIED = "verified"
    REPAIRED = "repaired"
    FAILED = "failed"


class ToolStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    """Represents a tool call made during reasoning"""
    tool_name: str
    tool_input: Dict[str, Any]
    raw_text: str = ""  # Original text containing the tool call
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'tool_input': self.tool_input,
            'raw_text': self.raw_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        return cls(
            tool_name=data.get('tool_name', ''),
            tool_input=data.get('tool_input', {}),
            raw_text=data.get('raw_text', '')
        )


@dataclass
class ToolOutput:
    """Represents the output from a tool execution"""
    tool_name: str
    result: str
    status: ToolStatus
    error: Optional[str] = None
    processed_images: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'result': self.result,
            'status': self.status.value,
            'error': self.error,
            'processed_images': self.processed_images,
            'execution_time_ms': self.execution_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolOutput':
        return cls(
            tool_name=data.get('tool_name', ''),
            result=data.get('result', ''),
            status=ToolStatus(data.get('status', 'success')),
            error=data.get('error'),
            processed_images=data.get('processed_images', []),
            execution_time_ms=data.get('execution_time_ms', 0.0)
        )


@dataclass
class VerificationResult:
    """Result from step verification"""
    step_index: int
    score: float  # -1 to 1, factual correctness
    confidence: float  # 0 to 1, verifier's certainty
    critique: str
    tool_check: bool = False  # Whether tool was used for verification
    raw_output: str = ""  # Raw verifier output
    
    @property
    def needs_repair(self) -> bool:
        """Check if step needs repair based on confidence threshold"""
        return self.confidence < 0.7
    
    @property
    def is_correct(self) -> bool:
        """Check if step is considered correct"""
        return self.score > 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_index': self.step_index,
            'score': self.score,
            'confidence': self.confidence,
            'critique': self.critique,
            'tool_check': self.tool_check,
            'raw_output': self.raw_output
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        return cls(
            step_index=data.get('step_index', 0),
            score=data.get('score', 0.0),
            confidence=data.get('confidence', 0.5),
            critique=data.get('critique', ''),
            tool_check=data.get('tool_check', False),
            raw_output=data.get('raw_output', '')
        )


@dataclass
class RepairInstruction:
    """Instruction for repairing a flawed step"""
    action: str  # "PATCH" or "NO_CHANGE"
    target_step: int
    patch_type: Optional[str] = None  # "text", "code", "tool_call", "parameter"
    new_content: Optional[str] = None
    justification: Optional[str] = None
    reason: Optional[str] = None  # For NO_CHANGE action
    raw_output: str = ""
    
    @property
    def should_patch(self) -> bool:
        return self.action == "PATCH"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'target_step': self.target_step,
            'patch_type': self.patch_type,
            'new_content': self.new_content,
            'justification': self.justification,
            'reason': self.reason,
            'raw_output': self.raw_output
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepairInstruction':
        return cls(
            action=data.get('action', 'NO_CHANGE'),
            target_step=data.get('target_step', 0),
            patch_type=data.get('patch_type'),
            new_content=data.get('new_content'),
            justification=data.get('justification'),
            reason=data.get('reason'),
            raw_output=data.get('raw_output', '')
        )


@dataclass
class ReasoningStep:
    """A single reasoning step in the trajectory"""
    step_index: int
    content: str  # The reasoning content (solver output)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_outputs: List[ToolOutput] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    repair: Optional[RepairInstruction] = None
    
    # Token positions for reward computation
    start_position: int = 0  # Token position where step starts
    end_position: int = 0  # Token position where step ends
    
    # Status tracking
    status: StepStatus = StepStatus.PENDING
    
    # Timing information
    generation_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    repair_time_ms: float = 0.0
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
    
    @property
    def all_tools_succeeded(self) -> bool:
        if not self.tool_outputs:
            return True
        return all(o.status == ToolStatus.SUCCESS for o in self.tool_outputs)
    
    @property
    def was_repaired(self) -> bool:
        return self.repair is not None and self.repair.should_patch
    
    @property
    def token_length(self) -> int:
        return self.end_position - self.start_position
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_index': self.step_index,
            'content': self.content,
            'tool_calls': [tc.to_dict() for tc in self.tool_calls],
            'tool_outputs': [to.to_dict() for to in self.tool_outputs],
            'verification': self.verification.to_dict() if self.verification else None,
            'repair': self.repair.to_dict() if self.repair else None,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'status': self.status.value,
            'generation_time_ms': self.generation_time_ms,
            'verification_time_ms': self.verification_time_ms,
            'repair_time_ms': self.repair_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        return cls(
            step_index=data.get('step_index', 0),
            content=data.get('content', ''),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get('tool_calls', [])],
            tool_outputs=[ToolOutput.from_dict(to) for to in data.get('tool_outputs', [])],
            verification=VerificationResult.from_dict(data['verification']) if data.get('verification') else None,
            repair=RepairInstruction.from_dict(data['repair']) if data.get('repair') else None,
            start_position=data.get('start_position', 0),
            end_position=data.get('end_position', 0),
            status=StepStatus(data.get('status', 'pending')),
            generation_time_ms=data.get('generation_time_ms', 0.0),
            verification_time_ms=data.get('verification_time_ms', 0.0),
            repair_time_ms=data.get('repair_time_ms', 0.0)
        )


@dataclass
class Agent0Trajectory:
    """Complete trajectory for Agent0-VL reasoning"""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: Optional[float] = None
    
    # Execution context for tool state persistence
    execution_context: Optional[Dict[str, Any]] = None
    
    # Metadata
    num_repairs: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    
    # Input information
    prompt: str = ""
    image_path: Optional[str] = None
    ground_truth: Optional[str] = None
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    
    @property
    def is_complete(self) -> bool:
        return self.final_answer is not None
    
    @property
    def all_steps_verified(self) -> bool:
        return all(s.verification is not None for s in self.steps)
    
    @property
    def average_confidence(self) -> float:
        if not self.steps:
            return 0.0
        confidences = [s.verification.confidence for s in self.steps if s.verification]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    @property
    def average_score(self) -> float:
        if not self.steps:
            return 0.0
        scores = [s.verification.score for s in self.steps if s.verification]
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def tool_success_rate(self) -> float:
        total_tools = sum(len(s.tool_outputs) for s in self.steps)
        if total_tools == 0:
            return 1.0
        successful = sum(
            1 for s in self.steps 
            for o in s.tool_outputs 
            if o.status == ToolStatus.SUCCESS
        )
        return successful / total_tools
    
    @property
    def repair_rate(self) -> float:
        if not self.steps:
            return 0.0
        repaired = sum(1 for s in self.steps if s.was_repaired)
        return repaired / len(self.steps)
    
    def add_step(self, step: ReasoningStep):
        """Add a new reasoning step"""
        self.steps.append(step)
        self.total_tokens = max(self.total_tokens, step.end_position)
    
    def get_step(self, index: int) -> Optional[ReasoningStep]:
        """Get step by index (1-based)"""
        for step in self.steps:
            if step.step_index == index:
                return step
        return None
    
    def get_last_step(self) -> Optional[ReasoningStep]:
        """Get the most recent step"""
        return self.steps[-1] if self.steps else None
    
    def get_step_boundaries(self) -> List[int]:
        """Get token positions where each step ends"""
        return [s.end_position for s in self.steps]
    
    def get_verification_scores(self) -> List[float]:
        """Get verification scores for all steps"""
        return [
            s.verification.score if s.verification else 0.0 
            for s in self.steps
        ]
    
    def get_verification_confidences(self) -> List[float]:
        """Get verification confidences for all steps"""
        return [
            s.verification.confidence if s.verification else 0.5 
            for s in self.steps
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'steps': [s.to_dict() for s in self.steps],
            'final_answer': self.final_answer,
            'confidence': self.confidence,
            'execution_context': self.execution_context,
            'num_repairs': self.num_repairs,
            'total_tokens': self.total_tokens,
            'total_time_ms': self.total_time_ms,
            'prompt': self.prompt,
            'image_path': self.image_path,
            'ground_truth': self.ground_truth
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent0Trajectory':
        return cls(
            steps=[ReasoningStep.from_dict(s) for s in data.get('steps', [])],
            final_answer=data.get('final_answer'),
            confidence=data.get('confidence'),
            execution_context=data.get('execution_context'),
            num_repairs=data.get('num_repairs', 0),
            total_tokens=data.get('total_tokens', 0),
            total_time_ms=data.get('total_time_ms', 0.0),
            prompt=data.get('prompt', ''),
            image_path=data.get('image_path'),
            ground_truth=data.get('ground_truth')
        )
    
    def to_json(self) -> str:
        """Serialize trajectory to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Agent0Trajectory':
        """Deserialize trajectory from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TrajectoryBatch:
    """Batch of trajectories for parallel processing"""
    trajectories: List[Agent0Trajectory] = field(default_factory=list)
    
    @property
    def batch_size(self) -> int:
        return len(self.trajectories)
    
    @property
    def all_complete(self) -> bool:
        return all(t.is_complete for t in self.trajectories)
    
    @property
    def completion_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(1 for t in self.trajectories if t.is_complete) / len(self.trajectories)
    
    def get_active_indices(self) -> List[int]:
        """Get indices of trajectories that are not yet complete"""
        return [i for i, t in enumerate(self.trajectories) if not t.is_complete]
    
    def get_final_answers(self) -> List[Optional[str]]:
        """Get final answers from all trajectories"""
        return [t.final_answer for t in self.trajectories]
    
    def get_confidences(self) -> List[float]:
        """Get confidence scores from all trajectories"""
        return [t.confidence if t.confidence is not None else 0.0 for t in self.trajectories]
    
    def get_num_steps(self) -> List[int]:
        """Get number of steps for each trajectory"""
        return [t.num_steps for t in self.trajectories]
    
    def get_num_repairs(self) -> List[int]:
        """Get number of repairs for each trajectory"""
        return [t.num_repairs for t in self.trajectories]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trajectories': [t.to_dict() for t in self.trajectories],
            'batch_size': self.batch_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryBatch':
        return cls(
            trajectories=[Agent0Trajectory.from_dict(t) for t in data.get('trajectories', [])]
        )


def create_trajectory_from_rollout_data(
    steps_data: List[Dict[str, Any]],
    final_answer: Optional[str] = None,
    confidence: Optional[float] = None,
    execution_context: Optional[Dict[str, Any]] = None
) -> Agent0Trajectory:
    """
    Create Agent0Trajectory from rollout worker output data
    
    Args:
        steps_data: List of step dictionaries from rollout
        final_answer: Extracted final answer
        confidence: Extracted confidence score
        execution_context: Tool execution context
    
    Returns:
        Agent0Trajectory instance
    """
    trajectory = Agent0Trajectory(
        final_answer=final_answer,
        confidence=confidence,
        execution_context=execution_context
    )
    
    for step_data in steps_data:
        # Parse tool calls
        tool_calls = []
        for tc in step_data.get('tool_calls', []):
            if isinstance(tc, dict):
                tool_calls.append(ToolCall(
                    tool_name=tc.get('tool_name', ''),
                    tool_input=tc.get('tool_input', {})
                ))
        
        # Parse tool outputs
        tool_outputs = []
        for to in step_data.get('tool_results', []):
            if isinstance(to, dict):
                tool_outputs.append(ToolOutput(
                    tool_name=to.get('tool_name', ''),
                    result=str(to.get('result', '')),
                    status=ToolStatus.SUCCESS if to.get('status') == 'success' else ToolStatus.ERROR,
                    error=to.get('error'),
                    processed_images=to.get('processed_images', [])
                ))
        
        # Parse verification
        verification = None
        if step_data.get('verification'):
            v = step_data['verification']
            verification = VerificationResult(
                step_index=v.get('step_index', step_data.get('step_index', 0)),
                score=v.get('score', 0.0),
                confidence=v.get('confidence', 0.5),
                critique=v.get('critique', ''),
                tool_check=v.get('tool_check', False)
            )
        
        # Parse repair
        repair = None
        if step_data.get('repair'):
            r = step_data['repair']
            repair = RepairInstruction(
                action=r.get('action', 'NO_CHANGE'),
                target_step=r.get('target_step', step_data.get('step_index', 0)),
                patch_type=r.get('patch_type'),
                new_content=r.get('new_content'),
                justification=r.get('justification'),
                reason=r.get('reason')
            )
        
        step = ReasoningStep(
            step_index=step_data.get('step_index', len(trajectory.steps) + 1),
            content=step_data.get('solver_text', ''),
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            verification=verification,
            repair=repair,
            start_position=step_data.get('start_pos', 0),
            end_position=step_data.get('end_pos', 0),
            status=StepStatus.COMPLETED
        )
        
        trajectory.add_step(step)
        
        if repair and repair.should_patch:
            trajectory.num_repairs += 1
    
    return trajectory
