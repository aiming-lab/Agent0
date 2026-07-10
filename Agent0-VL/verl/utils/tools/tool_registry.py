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
Tool registry and execution system for Agent0-VL
Provides unified interface for all external tools used during reasoning
"""

import ast
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np
from verl.utils.sandbox import execute_code_in_sandbox, TEMP_PROCESSED_IMAGES_DIR


class ToolExecutor:
    """Base class for tool executors"""

    def __init__(self):
        self.name = self.__class__.__name__

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given input

        Args:
            tool_input: Dictionary containing tool-specific parameters
            **kwargs: Additional context (image_path, previous_context, etc.)

        Returns:
            Dictionary with 'result', 'status', and optionally 'error'
        """
        raise NotImplementedError


class PythonExecTool(ToolExecutor):
    """Execute Python code for mathematical computation and symbolic reasoning"""

    def __init__(self):
        super().__init__()
        self.name = "PythonExec"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute Python code in sandbox

        Args:
            tool_input: {'code': str} - Python code to execute
            **kwargs: image_path, item_id, temp_dir, previous_context

        Returns:
            {'result': str, 'status': 'success'|'error', 'error': str (if failed)}
        """
        code = tool_input.get('code', '')
        if not code:
            return {'result': '', 'status': 'error', 'error': 'No code provided'}

        item_id = kwargs.get('item_id', 'N/A')
        image_path = kwargs.get('image_path', None)
        temp_dir = kwargs.get('temp_dir', TEMP_PROCESSED_IMAGES_DIR)
        previous_context = kwargs.get('previous_context', None)

        processed_paths, print_output, error_msg, execution_context = execute_code_in_sandbox(
            code_to_execute=code,
            input_image_path=image_path,
            item_id=item_id,
            temp_output_dir=temp_dir,
            previous_execution_context=previous_context
        )

        if error_msg:
            return {
                'result': print_output if print_output else '',
                'status': 'error',
                'error': error_msg,
                'execution_context': execution_context
            }

        result = {
            'result': print_output if print_output else 'Code executed successfully',
            'status': 'success',
            'processed_images': processed_paths,
            'execution_context': execution_context
        }

        return result


class OCRTool(ToolExecutor):
    """Extract text from figures using OCR.

    NOTE: This is an unimplemented stub. No OCR backend (e.g. EasyOCR,
    PaddleOCR) is wired in, so it never returns real text. It validates its
    input and then reports status 'unavailable' rather than fabricating a
    result, so callers can detect that OCR is not usable.
    """

    def __init__(self):
        super().__init__()
        self.name = "OCR"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Extract text from image region

        Args:
            tool_input: {'image_path': str, 'bbox': list (optional)}

        Returns:
            {'result': str, 'status': 'error'|'unavailable'}
        """
        image_path = tool_input.get('image_path')

        if not image_path:
            return {'result': '', 'status': 'error', 'error': 'No image_path provided'}

        return {
            'result': '',
            'status': 'unavailable',
            'error': 'OCR backend is not configured; no text was extracted.',
        }


class PlotParserTool(ToolExecutor):
    """Parse charts and extract numerical data.

    NOTE: This is an unimplemented stub. No chart-parsing backend is wired in,
    so it never returns real data. It validates its input and reports status
    'unavailable' rather than fabricating empty data points.
    """

    def __init__(self):
        super().__init__()
        self.name = "PlotParser"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Parse plot/chart and extract data

        Args:
            tool_input: {'image_path': str, 'parse_type': str}

        Returns:
            {'result': str, 'status': 'error'|'unavailable'}
        """
        image_path = tool_input.get('image_path')

        if not image_path:
            return {'result': '', 'status': 'error', 'error': 'No image_path provided'}

        return {
            'result': '',
            'status': 'unavailable',
            'error': 'Chart-parsing backend is not configured; no data was extracted.',
        }


class VisualAnalyzerTool(ToolExecutor):
    """Analyze image regions, colors, and spatial relationships"""

    def __init__(self):
        super().__init__()
        self.name = "VisualAnalyzer"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze visual properties

        Args:
            tool_input: {'image_path': str, 'analysis_type': str, 'region': list (optional)}

        Returns:
            {'result': dict, 'status': 'success'|'error'}
        """
        try:
            image_path = tool_input.get('image_path')
            analysis_type = tool_input.get('analysis_type', 'color')
            region = tool_input.get('region', None)

            if not image_path:
                return {'result': {}, 'status': 'error', 'error': 'No image_path provided'}

            # Load image (context manager ensures the file handle is released)
            with Image.open(image_path) as img:
                width, height = img.width, img.height
                img_array = np.array(img)

            if region:
                x1, y1, x2, y2 = region
                img_array = img_array[y1:y2, x1:x2]

            # Perform analysis based on type
            if analysis_type == 'color':
                # Get dominant color
                mean_color = img_array.mean(axis=(0, 1)).tolist()
                result = {'dominant_color': mean_color, 'analysis_type': 'color'}
            elif analysis_type == 'size':
                result = {'width': width, 'height': height, 'analysis_type': 'size'}
            else:
                result = {'analysis_type': analysis_type, 'info': 'Analysis completed'}

            return {'result': json.dumps(result), 'status': 'success'}

        except Exception as e:
            return {'result': {}, 'status': 'error', 'error': str(e)}


class DetectorTool(ToolExecutor):
    """Object detection with bounding boxes.

    NOTE: This is an unimplemented stub. No detection model is wired in, so it
    never returns real detections. It validates its input and reports status
    'unavailable' rather than fabricating empty detection results.
    """

    def __init__(self):
        super().__init__()
        self.name = "Detector"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Detect objects in image

        Args:
            tool_input: {'image_path': str, 'objects': list (optional)}

        Returns:
            {'result': str, 'status': 'error'|'unavailable'}
        """
        image_path = tool_input.get('image_path')

        if not image_path:
            return {'result': '', 'status': 'error', 'error': 'No image_path provided'}

        return {
            'result': '',
            'status': 'unavailable',
            'error': 'Object-detection backend is not configured; no objects were detected.',
        }


class RetrieverTool(ToolExecutor):
    """Knowledge lookup and retrieval.

    NOTE: This is an unimplemented stub. No knowledge base or search API is
    wired in, so it never returns real information. It validates its input and
    reports status 'unavailable' rather than fabricating a retrieved snippet.
    """

    def __init__(self):
        super().__init__()
        self.name = "Retriever"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Retrieve knowledge

        Args:
            tool_input: {'query': str, 'source': str (optional)}

        Returns:
            {'result': str, 'status': 'error'|'unavailable'}
        """
        query = tool_input.get('query', '')

        if not query:
            return {'result': '', 'status': 'error', 'error': 'No query provided'}

        return {
            'result': '',
            'status': 'unavailable',
            'error': 'Retrieval backend is not configured; no information was retrieved.',
        }


class ImageToolExecutor(ToolExecutor):
    """Image manipulation tools (crop, zoom, etc.)"""

    def __init__(self):
        super().__init__()
        self.name = "ImageTool"

    def execute(self, tool_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform image manipulation

        Args:
            tool_input: {
                'operation': str ('crop'|'zoom'|'rotate'),
                'image_path': str,
                'params': dict
            }

        Returns:
            {'result': str (output path), 'status': 'success'|'error'}
        """
        try:
            operation = tool_input.get('operation', 'crop')
            image_path = tool_input.get('image_path')
            params = tool_input.get('params', {})

            if not image_path:
                return {'result': '', 'status': 'error', 'error': 'No image_path provided'}

            # Context manager releases the source file handle; each branch
            # produces a new image object that outlives the `with` block.
            with Image.open(image_path) as img:
                if operation == 'crop':
                    bbox = params.get('bbox', [0, 0, img.width, img.height])
                    img_processed = img.crop(tuple(bbox))
                elif operation == 'zoom':
                    scale = params.get('scale', 1.0)
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img_processed = img.resize(new_size)
                elif operation == 'rotate':
                    angle = params.get('angle', 0)
                    img_processed = img.rotate(angle)
                else:
                    return {'result': '', 'status': 'error', 'error': f'Unknown operation: {operation}'}

                # crop() is lazy and references the source image; force it to
                # materialize before the source file handle is closed.
                img_processed.load()

            # Save processed image
            temp_dir = kwargs.get('temp_dir', TEMP_PROCESSED_IMAGES_DIR)
            import os
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f'processed_{operation}_{os.path.basename(image_path)}')
            img_processed.save(output_path)

            return {'result': output_path, 'status': 'success'}

        except Exception as e:
            return {'result': '', 'status': 'error', 'error': str(e)}


class ToolRegistry:
    """Central registry for all tools"""

    def __init__(self):
        self.tools: Dict[str, ToolExecutor] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default tools"""
        self.register_tool(PythonExecTool())
        self.register_tool(OCRTool())
        self.register_tool(PlotParserTool())
        self.register_tool(VisualAnalyzerTool())
        self.register_tool(DetectorTool())
        self.register_tool(RetrieverTool())
        self.register_tool(ImageToolExecutor())

    def register_tool(self, tool: ToolExecutor):
        """Register a new tool"""
        self.tools[tool.name] = tool

    def get_tool(self, tool_name: str) -> Optional[ToolExecutor]:
        """Get tool by name"""
        return self.tools.get(tool_name)

    def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            **kwargs: Additional context

        Returns:
            Tool execution result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                'result': '',
                'status': 'error',
                'error': f'Tool not found: {tool_name}'
            }

        try:
            return tool.execute(tool_input, **kwargs)
        except Exception as e:
            return {
                'result': '',
                'status': 'error',
                'error': f'Tool execution failed: {str(e)}'
            }

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for tool"""
        tool_schemas = {
            'PythonExec': {
                'name': 'PythonExec',
                'description': 'Execute Python code for mathematical computation and symbolic reasoning',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'code': {'type': 'string', 'description': 'Python code to execute'}
                    },
                    'required': ['code']
                }
            },
            'OCR': {
                'name': 'OCR',
                'description': 'Extract text from images using OCR',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {'type': 'string', 'description': 'Path to image file'},
                        'bbox': {'type': 'array', 'description': 'Bounding box [x1, y1, x2, y2] (optional)'}
                    },
                    'required': ['image_path']
                }
            },
            'PlotParser': {
                'name': 'PlotParser',
                'description': 'Parse charts and extract numerical data',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {'type': 'string', 'description': 'Path to chart image'},
                        'parse_type': {'type': 'string', 'description': 'Type of chart (bar, line, pie, etc.)'}
                    },
                    'required': ['image_path']
                }
            },
            'VisualAnalyzer': {
                'name': 'VisualAnalyzer',
                'description': 'Analyze image regions, colors, and spatial relationships',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {'type': 'string'},
                        'analysis_type': {'type': 'string', 'description': 'color, size, spatial'},
                        'region': {'type': 'array', 'description': '[x1, y1, x2, y2] (optional)'}
                    },
                    'required': ['image_path']
                }
            },
            'Detector': {
                'name': 'Detector',
                'description': 'Detect objects in image with bounding boxes',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {'type': 'string'},
                        'objects': {'type': 'array', 'description': 'List of objects to detect (optional)'}
                    },
                    'required': ['image_path']
                }
            },
            'Retriever': {
                'name': 'Retriever',
                'description': 'Retrieve knowledge and information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Query string'},
                        'source': {'type': 'string', 'description': 'Knowledge source (optional)'}
                    },
                    'required': ['query']
                }
            },
            'ImageTool': {
                'name': 'ImageTool',
                'description': 'Image manipulation (crop, zoom, rotate)',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'operation': {'type': 'string', 'description': 'crop, zoom, rotate'},
                        'image_path': {'type': 'string'},
                        'params': {'type': 'object', 'description': 'Operation-specific parameters'}
                    },
                    'required': ['operation', 'image_path']
                }
            }
        }

        return tool_schemas.get(tool_name)


def _find_balanced_object(text: str, start: int) -> Optional[str]:
    """Return the substring of a balanced ``{...}`` object beginning at ``start``.

    Scans from the opening brace at ``text[start]`` and tracks brace depth while
    respecting string literals (so braces inside quoted strings are ignored).
    Returns ``None`` if no balanced object is found.
    """
    if start >= len(text) or text[start] != '{':
        return None

    depth = 0
    in_string = False
    quote_char = ''
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _loads_flexible(obj_str: str) -> Optional[Any]:
    """Parse a JSON-ish object string, tolerating single-quoted Python dicts.

    Tries strict ``json.loads`` first, then falls back to ``ast.literal_eval``
    for single-quoted dict literals (as some models emit). Returns ``None`` on
    failure.
    """
    try:
        return json.loads(obj_str)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        value = ast.literal_eval(obj_str)
        if isinstance(value, dict):
            return value
    except (ValueError, SyntaxError):
        pass
    return None


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse tool call from model output.

    Uses a brace-matching scanner so that tool calls with nested ``tool_input``
    objects (e.g. ImageTool's ``{"params": {"bbox": [...]}}``) are extracted
    correctly, whether they are standalone or embedded in surrounding prose.
    Both double-quoted JSON and single-quoted Python dict literals are accepted.

    Expected format:
    {"tool_name":"PythonExec","tool_input":{"code":"..."}}

    Args:
        text: Model output text

    Returns:
        Tuple of (tool_name, tool_input) or None if no valid tool call found
    """
    if not text:
        return None

    # Locate every candidate object that opens with a "tool_name" key, allowing
    # arbitrary whitespace and either quote style.
    key_pattern = re.compile(r'\{\s*["\']tool_name["\']')

    for match in key_pattern.finditer(text):
        obj_str = _find_balanced_object(text, match.start())
        if obj_str is None:
            continue

        data = _loads_flexible(obj_str)
        if isinstance(data, dict) and 'tool_name' in data and 'tool_input' in data:
            tool_input = data['tool_input']
            if isinstance(tool_input, dict):
                return data['tool_name'], tool_input

    # Fallback: the whole text may itself be a single JSON/dict object.
    data = _loads_flexible(text.strip())
    if isinstance(data, dict) and 'tool_name' in data and 'tool_input' in data:
        tool_input = data['tool_input']
        if isinstance(tool_input, dict):
            return data['tool_name'], tool_input

    return None


# Global tool registry instance
_global_tool_registry = None


def get_global_tool_registry() -> ToolRegistry:
    """Get or create global tool registry instance"""
    global _global_tool_registry
    if _global_tool_registry is None:
        _global_tool_registry = ToolRegistry()
    return _global_tool_registry
