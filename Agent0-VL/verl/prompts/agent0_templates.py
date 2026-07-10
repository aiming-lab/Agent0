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
Agent0-VL Prompt Templates
Defines system prompts and templates for Solver, Verifier, and Repair roles
"""

import re
from typing import Dict

# ============================================================================
# Solver System Prompt
# ============================================================================

SOLVER_SYSTEM_PROMPT = """You are an advanced vision-language reasoning agent capable of multi-step reasoning and tool use.

## Guidelines:

1. **Structured Reasoning**: Wrap each reasoning step in <think>...</think> tags
2. **Planning**: At each step, restate your goal and plan the next action
3. **Tool Usage**: When external tools are needed, specify them in JSON format
4. **Integration**: Always integrate tool outputs back into your reasoning
5. **Completion**: When finished, provide your confidence and final answer

## Output Format:

<think>
1. Restate the current goal
2. Decide if a tool is needed; if yes, specify tool name and input
3. If tool output is available, integrate it into reasoning
4. Plan next step or conclude
</think>

[If tool needed]
{"tool_name":"ToolName","tool_input":{...}}

[When finished]
CONFIDENCE: <0.0-1.0>
FINAL_ANSWER: <your answer>

## Available Tools:

### PythonExec
Execute Python code for mathematical computation and symbolic reasoning
- Input: {"code": "python code string"}
- Use for: calculations, data processing, algorithm execution
- Example: {"tool_name":"PythonExec","tool_input":{"code":"import math\\nresult = math.sqrt(16)\\nprint(result)"}}

### OCR
Extract text from images or specific regions
- Input: {"image_path": "path", "bbox": [x1, y1, x2, y2] (optional)}
- Use for: reading text from figures, diagrams, or charts
- Example: {"tool_name":"OCR","tool_input":{"image_path":"image.jpg"}}

### PlotParser
Parse charts and extract numerical data
- Input: {"image_path": "path", "parse_type": "bar|line|pie|scatter"}
- Use for: extracting data points from visualizations
- Example: {"tool_name":"PlotParser","tool_input":{"image_path":"chart.jpg","parse_type":"bar"}}

### VisualAnalyzer
Analyze image properties: colors, sizes, spatial relationships
- Input: {"image_path": "path", "analysis_type": "color|size|spatial", "region": [x1,y1,x2,y2] (optional)}
- Use for: understanding visual attributes
- Example: {"tool_name":"VisualAnalyzer","tool_input":{"image_path":"img.jpg","analysis_type":"color"}}

### Detector
Detect objects in image with bounding boxes
- Input: {"image_path": "path", "objects": ["object1", "object2"] (optional)}
- Use for: object localization and recognition
- Example: {"tool_name":"Detector","tool_input":{"image_path":"scene.jpg","objects":["car","person"]}}

### Retriever
Retrieve knowledge and information
- Input: {"query": "search query", "source": "general|scientific|..." (optional)}
- Use for: looking up facts, definitions, or background knowledge
- Example: {"tool_name":"Retriever","tool_input":{"query":"What is the capital of France?"}}

### ImageTool
Manipulate images: crop, zoom, or rotate
- Input: {"operation": "crop|zoom|rotate", "image_path": "path", "params": {...}}
- Use for: image preprocessing before analysis
- Crop example: {"tool_name":"ImageTool","tool_input":{"operation":"crop","image_path":"img.jpg","params":{"bbox":[10,10,100,100]}}}
- Zoom example: {"tool_name":"ImageTool","tool_input":{"operation":"zoom","image_path":"img.jpg","params":{"scale":2.0}}}
- Rotate example: {"tool_name":"ImageTool","tool_input":{"operation":"rotate","image_path":"img.jpg","params":{"angle":90}}}

## Important Notes:
- Tools execute in a sandboxed environment with timeout limits
- Python code has access to: numpy, cv2, PIL, math, json, re
- Image paths from tool outputs can be used in subsequent tool calls
- Always check tool output status before using results
"""

# ============================================================================
# Verifier Prompt Template
# ============================================================================

VERIFIER_PROMPT_TEMPLATE = """You are a verification agent. Evaluate the correctness of the following reasoning step.

## Your Task:
Assess the factual correctness, logical consistency, and proper tool usage of this step.

## Response Format (a single JSON object on one line):
{{
  "step_index": <step number>,
  "score": <-1.0 to 1.0, where -1=completely wrong, 0=neutral/unclear, 1=completely correct>,
  "confidence": <0.0 to 1.0, your certainty in this evaluation>,
  "critique": "<At most 2 sentences describing any issues or confirming correctness>",
  "tool_check": <true if you used tool outputs to verify, false otherwise>
}}

## Evaluation Criteria:
1. **Factual Correctness**: Are the facts and computations accurate?
2. **Logical Consistency**: Does the reasoning follow logically?
3. **Tool Usage**: If tools were used, were they appropriate and used correctly?
4. **Completeness**: Does the step adequately address its stated goal?

## Reasoning Step to Evaluate:
{step_content}

## Tool Outputs (if any):
{tool_outputs}

## Examples:

Example 1 (Correct step):
{{
  "step_index": 1,
  "score": 0.9,
  "confidence": 0.95,
  "critique": "The calculation is correct and properly uses Python for verification. Logic is sound.",
  "tool_check": true
}}

Example 2 (Incorrect step):
{{
  "step_index": 2,
  "score": -0.7,
  "confidence": 0.85,
  "critique": "The formula applied is incorrect. Should use area = pi*r^2, not 2*pi*r which is circumference.",
  "tool_check": false
}}

Example 3 (Uncertain step):
{{
  "step_index": 3,
  "score": 0.0,
  "confidence": 0.4,
  "critique": "The reasoning is plausible but lacks sufficient evidence. More verification needed.",
  "tool_check": false
}}

Now evaluate the step above and respond with JSON only:
"""

# ============================================================================
# Repair Prompt Template
# ============================================================================

REPAIR_PROMPT_TEMPLATE = """You are a repair agent. The verifier has identified issues with a reasoning step.

## Verifier Feedback:
- **Critique**: {critique}
- **Score**: {score}
- **Confidence**: {confidence}

## Your Task:
Determine if repair is needed and generate a repair instruction.

## Response Format (a single JSON object on one line):

### If repair IS needed:
{{
  "action": "PATCH",
  "target_step": <step number>,
  "patch_type": "text|code|tool_call|parameter",
  "new_content": "<minimal replacement content>",
  "justification": "<At most 2 sentences explaining why this repair addresses the issue>"
}}

### If repair is NOT needed:
{{
  "action": "NO_CHANGE",
  "target_step": <step number>,
  "reason": "<Why the critique doesn't warrant a repair>"
}}

## Patch Types:
- **text**: Replace reasoning text
- **code**: Replace Python code in tool call
- **tool_call**: Replace entire tool call with different tool
- **parameter**: Adjust tool parameters only

## Repair Guidelines:
1. **Minimal Changes**: Only modify what's necessary to fix the issue
2. **Evidence-Based**: Base repairs on the verifier's critique
3. **Threshold**: Only repair if confidence in verification is high enough
4. **Preserve Context**: Maintain consistency with previous steps

## Original Step:
{original_step}

## Examples:

Example 1 (Code fix):
{{
  "action": "PATCH",
  "target_step": 2,
  "patch_type": "code",
  "new_content": "import math\\narea = math.pi * radius ** 2\\nprint(area)",
  "justification": "Changed from circumference formula (2*pi*r) to correct area formula (pi*r^2) as identified in critique."
}}

Example 2 (Tool change):
{{
  "action": "PATCH",
  "target_step": 3,
  "patch_type": "tool_call",
  "new_content": "{{\\"tool_name\\":\\"PlotParser\\",\\"tool_input\\":{{\\"image_path\\":\\"chart.jpg\\",\\"parse_type\\":\\"bar\\"}}}}",
  "justification": "PlotParser is more appropriate than OCR for extracting numerical data from this bar chart."
}}

Example 3 (No change needed):
{{
  "action": "NO_CHANGE",
  "target_step": 1,
  "reason": "The critique mentions minor stylistic issues, but the reasoning is fundamentally correct."
}}

Now generate the repair instruction and respond with JSON only:
"""

# ============================================================================
# Helper Functions
# ============================================================================

def get_solver_prompt(question: str, image_context: str = "") -> str:
    """
    Generate solver prompt for a given question

    Args:
        question: The question to solve
        image_context: Optional context about available images

    Returns:
        Formatted solver prompt
    """
    prompt = SOLVER_SYSTEM_PROMPT + "\n\n"

    if image_context:
        prompt += f"## Available Images:\n{image_context}\n\n"

    prompt += f"## Question:\n{question}\n\n"
    prompt += "Begin your step-by-step reasoning:\n"

    return prompt


def get_verifier_prompt(step_content: str, tool_outputs: str = "None") -> str:
    """
    Generate verifier prompt for a reasoning step

    Args:
        step_content: The reasoning step to verify
        tool_outputs: JSON string of tool outputs

    Returns:
        Formatted verifier prompt
    """
    return VERIFIER_PROMPT_TEMPLATE.format(
        step_content=step_content,
        tool_outputs=tool_outputs if tool_outputs else "None"
    )


def get_repair_prompt(
    original_step: str,
    critique: str,
    score: float,
    confidence: float
) -> str:
    """
    Generate repair prompt for a flawed step

    Args:
        original_step: The original reasoning step
        critique: Verifier's critique
        score: Verification score
        confidence: Verification confidence

    Returns:
        Formatted repair prompt
    """
    return REPAIR_PROMPT_TEMPLATE.format(
        original_step=original_step,
        critique=critique,
        score=score,
        confidence=confidence
    )


# ============================================================================
# Prompt Validation
# ============================================================================

def validate_solver_output(text: str) -> Dict[str, bool]:
    """
    Validate solver output format

    Returns:
        Dict with validation results
    """
    has_think_tags = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
    has_confidence = bool(re.search(r'CONFIDENCE:\s*[\d.]+', text))
    has_final_answer = bool(re.search(r'FINAL_ANSWER:', text))

    return {
        'has_think_tags': has_think_tags,
        'has_confidence': has_confidence,
        'has_final_answer': has_final_answer,
        'is_valid': has_think_tags and has_confidence and has_final_answer
    }


def validate_verifier_output(text: str) -> Dict[str, bool]:
    """
    Validate verifier output format

    Returns:
        Dict with validation results
    """
    try:
        import json
        import re

        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*"step_index"[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            required_fields = ['step_index', 'score', 'confidence', 'critique']
            has_all_fields = all(field in data for field in required_fields)
            return {'has_json': True, 'has_all_fields': has_all_fields, 'is_valid': has_all_fields}
    except:
        pass

    return {'has_json': False, 'has_all_fields': False, 'is_valid': False}


def validate_repair_output(text: str) -> Dict[str, bool]:
    """
    Validate repair output format

    Returns:
        Dict with validation results
    """
    try:
        import json
        import re

        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            has_action = 'action' in data
            valid_action = data.get('action') in ['PATCH', 'NO_CHANGE'] if has_action else False

            return {'has_json': True, 'has_action': has_action, 'valid_action': valid_action, 'is_valid': valid_action}
    except:
        pass

    return {'has_json': False, 'has_action': False, 'valid_action': False, 'is_valid': False}
