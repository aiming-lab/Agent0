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
Sandbox code execution module for Agent0-VL
Provides secure Python code execution with timeout and resource limits
"""

import json
import os
import re
import contextlib
import ast
import sys
import io
import stat
import random
import string
import shutil
from PIL import Image, UnidentifiedImageError
import numpy
import math
import functools
import threading
import glob
import importlib
import types
import pickle
import textwrap
from typing import Dict, List, Tuple, Optional, Any

try:
    import autopep8
except ImportError:
    autopep8 = None

try:
    import cv2
except ImportError:
    cv2 = None


# Timeout support. ``timeout_decorator`` is preferred, but it is an optional
# dependency: when it is not installed we fall back to a thread-based timeout so
# the module (and its tests) work without extra packages.
try:
    import timeout_decorator

    class SandboxTimeoutError(timeout_decorator.TimeoutError):
        """Single timeout type callers can catch regardless of backend."""

    def _timeout(seconds):
        # use_signals=False -> thread/process based; safe to call off the main
        # thread (signal-based timeouts only work on the main thread).
        return timeout_decorator.timeout(
            seconds, use_signals=False, timeout_exception=SandboxTimeoutError
        )

except ImportError:  # pragma: no cover - exercised only when dep is absent
    timeout_decorator = None

    class SandboxTimeoutError(Exception):
        """Raised when sandboxed execution exceeds its time budget."""

    def _timeout(seconds):
        """Thread-based timeout fallback (subprocess-free, off-main-thread safe).

        A timed-out worker thread cannot be forcibly killed in CPython, so it is
        abandoned as a daemon thread while the caller observes a timeout. This is
        acceptable for the evaluator's short image-processing snippets.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                box = {}

                def target():
                    try:
                        box["result"] = func(*args, **kwargs)
                    except BaseException as exc:  # noqa: BLE001 - propagated below
                        box["error"] = exc

                worker = threading.Thread(target=target, daemon=True)
                worker.start()
                worker.join(seconds)
                if worker.is_alive():
                    raise SandboxTimeoutError(
                        f"Execution timed out after {seconds} seconds"
                    )
                if "error" in box:
                    raise box["error"]
                return box.get("result")

            return wrapper

        return decorator


# Security: Prohibit dangerous system calls. Static regex screening is a
# best-effort first line of defence (it is bypassable in principle), backed up
# by a restricted set of builtins/modules injected into the exec namespace.
DANGEROUS_PATTERNS = [
    r'\bsys\.',
    r'\bsocket\.',
    r'\bsubprocess\b',
    r'\bexec\(',
    r'\beval\(',
    r'\bcompile\(',
    r'\b__import__\(',
    r'\bos\.(remove|unlink|rmdir)\b',
    r'\bos\.system\b',
    r'\bos\.popen\b',
    r'\bos\.exec[a-z]*\b',   # execl, execv, execve, execvp, ...
    r'\bos\.spawn[a-z]*\b',  # spawnl, spawnv, spawnve, ...
    r'\bos\.fork\b',
    r'\bos\.kill\b',
    r'\bos\.putenv\b',
    r'\bshutil\.rmtree\b',
    r'\bshutil\.move\b',
    r'\bos\.(rename|renames)\b',
]

# Execution time limit (seconds)
EXEC_TIME_LIMIT = 10

# Temporary directory for processed images
TEMP_PROCESSED_IMAGES_DIR = "./temp_processed_images/"

# Minimum crop dimension
MIN_CROP_DIMENSION = 64

# Valid image extensions
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')


def check_dangerous_code(code_string: str) -> bool:
    """
    Performs static analysis to detect dangerous patterns in code

    Args:
        code_string: Python code string to check

    Returns:
        True if dangerous patterns found, False otherwise
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code_string, re.IGNORECASE):
            return True
    return False


class ReadOnlyPath:
    """
    Context manager to temporarily make files read-only during execution
    Prevents sandbox code from modifying original input files
    """

    def __init__(self, path: Optional[str]):
        self.path = path if isinstance(path, str) else None
        self.original_permissions = None

    def __enter__(self):
        """Save original permissions and make file read-only"""
        if self.path and os.path.isfile(self.path):
            try:
                self.original_permissions = os.stat(self.path).st_mode
                read_only_permissions = self.original_permissions & ~(
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                )
                if self.original_permissions != read_only_permissions:
                    os.chmod(self.path, read_only_permissions)
            except OSError as e:
                print(f"Warning: Could not make '{self.path}' read-only: {e}", file=sys.stderr)
                self.original_permissions = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original file permissions"""
        if self.path and self.original_permissions is not None and os.path.isfile(self.path):
            try:
                os.chmod(self.path, self.original_permissions)
            except OSError as e:
                print(f"Warning: Could not restore permissions for '{self.path}': {e}", file=sys.stderr)


class ImagePathTransformer(ast.NodeTransformer):
    """AST transformer to fix invalid image paths in generated code"""

    def __init__(self, replacement_path: str):
        self.replacement_path = replacement_path
        self.path_was_replaced = False

    def visit_Assign(self, node):
        """Process simple assignments: image_path = "string_literal" """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_variable_name = node.targets[0].id

            if target_variable_name == 'image_path':
                current_path_value = None
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    current_path_value = node.value.value
                elif isinstance(node.value, ast.Str):  # Python < 3.8
                    current_path_value = node.value.s

                if current_path_value is not None:
                    is_valid_image = (
                        os.path.exists(current_path_value) and
                        os.path.isfile(current_path_value) and
                        any(current_path_value.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)
                    )

                    if not is_valid_image:
                        if hasattr(ast, 'Constant'):  # Python 3.8+
                            node.value = ast.Constant(value=self.replacement_path)
                        else:
                            node.value = ast.Str(s=self.replacement_path)
                        self.path_was_replaced = True

        return self.generic_visit(node)


class CropCoordinateTransformer(ast.NodeTransformer):
    """AST transformer to clamp crop coordinates to image boundaries"""

    def __init__(self, image_width: int, image_height: int, main_image_variable_name: str = "image"):
        self.image_width = image_width
        self.image_height = image_height
        self.main_image_variable_name = main_image_variable_name
        self.coordinates_clamped = False
        self.known_coord_var_names = [
            'crop_box', 'bbox', 'box', 'coordinates', 'coords', 'crop_coords',
            'left_crop_coords', 'right_crop_coords', 'top_crop_coords', 'bottom_crop_coords',
        ]

    def _get_numeric_value(self, node_value):
        """Extract numeric value from AST node"""
        if isinstance(node_value, ast.Constant) and isinstance(node_value.value, (int, float)):
            return node_value.value
        elif isinstance(node_value, ast.Num):  # Python < 3.8
            return node_value.n
        return None

    def _create_numeric_node(self, value):
        """Create AST node for numeric value"""
        int_value = int(round(value))
        if hasattr(ast, 'Constant'):  # Python 3.8+
            return ast.Constant(value=int_value)
        else:
            return ast.Num(n=int_value)

    def _clamp_coordinates(self, v_x1, v_y1, v_x2, v_y2):
        """Clamp coordinates to image boundaries with minimum dimension constraints"""
        img_w, img_h = float(self.image_width), float(self.image_height)

        cx1 = min(v_x1, v_x2)
        cx2 = max(v_x1, v_x2)
        cy1 = min(v_y1, v_y2)
        cy2 = max(v_y1, v_y2)

        # Clamp to image boundaries
        final_x1 = max(0.0, min(cx1, img_w - 1.0 if img_w > 0 else 0.0))
        final_x2 = max(0.0, min(cx2, img_w))
        final_y1 = max(0.0, min(cy1, img_h - 1.0 if img_h > 0 else 0.0))
        final_y2 = max(0.0, min(cy2, img_h))

        # Enforce minimum dimensions
        current_width = final_x2 - final_x1
        if img_w > 0 and current_width < MIN_CROP_DIMENSION:
            if img_w < MIN_CROP_DIMENSION:
                final_x1, final_x2 = 0.0, img_w
            else:
                final_x2 = min(final_x1 + MIN_CROP_DIMENSION, img_w)
                if final_x2 > img_w:
                    final_x2 = img_w
                    final_x1 = max(0.0, final_x2 - MIN_CROP_DIMENSION)

        current_height = final_y2 - final_y1
        if img_h > 0 and current_height < MIN_CROP_DIMENSION:
            if img_h < MIN_CROP_DIMENSION:
                final_y1, final_y2 = 0.0, img_h
            else:
                final_y2 = min(final_y1 + MIN_CROP_DIMENSION, img_h)
                if final_y2 > img_h:
                    final_y2 = img_h
                    final_y1 = max(0.0, final_y2 - MIN_CROP_DIMENSION)

        return [final_x1, final_y1, final_x2, final_y2]

    def visit_Assign(self, node):
        """Visit assignment nodes to clamp coordinate tuples"""
        # Handle tuple assignments like: crop_box = (x1, y1, x2, y2)
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in self.known_coord_var_names and
                isinstance(node.value, ast.Tuple) and len(node.value.elts) == 4):

            raw_coords = [self._get_numeric_value(v) for v in node.value.elts]

            if all(c is not None for c in raw_coords):
                x1, y1, x2, y2 = raw_coords
                clamped = self._clamp_coordinates(x1, y1, x2, y2)

                original_rounded = [int(round(c)) for c in raw_coords]
                final_rounded = [int(round(c)) for c in clamped]

                if final_rounded != original_rounded:
                    self.coordinates_clamped = True
                    for i in range(4):
                        node.value.elts[i] = self._create_numeric_node(clamped[i])

        return self.generic_visit(node)


class OpenCVNamespaceTransformer(ast.NodeTransformer):
    """Transform incorrect OpenCV namespace calls (cv., cv4.) to cv2."""

    def __init__(self, incorrect_prefixes=None, correct_prefix="cv2"):
        if incorrect_prefixes is None:
            self.incorrect_prefixes = ["cv", "cv4", "cv3", 'CV']
        else:
            self.incorrect_prefixes = incorrect_prefixes
        self.correct_prefix = correct_prefix
        self.namespace_updated = False

    def visit_Attribute(self, node):
        """Replace incorrect cv namespace with cv2"""
        if isinstance(node.value, ast.Name) and node.value.id in self.incorrect_prefixes:
            node.value.id = self.correct_prefix
            self.namespace_updated = True
        return self.generic_visit(node)


def align_first_line_to_second(code_string: str) -> str:
    """
    Align indentation of first non-empty line to second non-empty line

    Args:
        code_string: Input code string

    Returns:
        Code string with fixed indentation
    """
    lines = code_string.splitlines()

    first_line_info = None
    second_line_info = None

    for index, line_content in enumerate(lines):
        if line_content.strip():
            if first_line_info is None:
                first_line_info = {'index': index, 'content': line_content}
            elif second_line_info is None:
                second_line_info = {'index': index, 'content': line_content}
                break

    if not first_line_info or not second_line_info:
        return code_string

    first_line_content = first_line_info['content']
    second_line_content = second_line_info['content']

    first_line_indent = ' ' * (len(first_line_content) - len(first_line_content.lstrip(' ')))
    second_line_indent = ' ' * (len(second_line_content) - len(second_line_content.lstrip(' ')))

    if first_line_indent != second_line_indent:
        original_index = first_line_info['index']
        stripped_content = first_line_content.lstrip(' ')
        lines[original_index] = second_line_indent + stripped_content

    return "\n".join(lines)


def get_image_paths(temp_output_dir: str) -> List[str]:
    """Get all image file paths in directory"""
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']
    image_paths = []

    for ext in extensions:
        pattern = os.path.join(temp_output_dir, f'*.{ext}')
        image_paths.extend(glob.glob(pattern))

    return image_paths


def ensure_temp_dir(temp_output_dir: str):
    """Ensure temporary directory exists and is writable"""
    os.makedirs(temp_output_dir, exist_ok=True)
    if not os.access(temp_output_dir, os.W_OK):
        raise PermissionError(f"Temporary directory {temp_output_dir} is not writable.")


def _make_restricted_os():
    """Build a restricted ``os``-like namespace for the sandbox.

    Exposes only the path/query helpers legitimate image code needs
    (``os.path`` plus ``getcwd``/``listdir``/``makedirs``/``environ.get``) and
    withholds the process-control surface (``system``/``popen``/``exec*``/
    ``spawn*``/``fork``/``kill``) that a full ``os`` module would grant.
    """
    restricted = types.SimpleNamespace(
        path=os.path,
        sep=os.sep,
        linesep=os.linesep,
        getcwd=os.getcwd,
        listdir=os.listdir,
        makedirs=os.makedirs,
        environ=types.SimpleNamespace(get=os.environ.get),
    )
    return restricted


@_timeout(EXEC_TIME_LIMIT)
def _sandboxed_execution_target(
    code_to_execute: str,
    input_image_path: Optional[str],
    temp_output_dir: str,
    item_id: str,
    previous_execution_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Target function for sandboxed code execution

    Args:
        code_to_execute: Python code string to execute
        input_image_path: Path to input image
        temp_output_dir: Directory for processed images
        item_id: Identifier for logging
        previous_execution_context: Context from previous execution step

    Returns:
        Dictionary with processed_paths_list, print_output, error, execution_context
    """
    return_dict = {}

    # A restricted os namespace shared by builtins and globals: legitimate
    # path/dir helpers are available, but process-control calls are not.
    restricted_os = _make_restricted_os()

    # Prepare restricted environment
    allowed_builtins = {
        "print": print, "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple, "range": range, "round": round,
        "abs": abs, "min": min, "max": max, "sum": sum, "sorted": sorted,
        "any": any, "all": all, "zip": zip, "map": map, "filter": filter,
        "True": True, "False": False, "None": None, "isinstance": isinstance,
        "issubclass": issubclass,
        # Common container / conversion builtins
        "set": set, "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
        "type": type, "repr": repr, "format": format, "chr": chr, "ord": ord,
        "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "callable": callable,
        "iter": iter, "next": next, "slice": slice, "object": object,
        "enumerate": enumerate, "pow": pow, "divmod": divmod, "bin": bin,
        "oct": oct, "hex": hex, "complex": complex, "__import__": __import__,
        # Exceptions
        "Exception": Exception, "BaseException": BaseException, "ValueError": ValueError,
        "TypeError": TypeError, "AttributeError": AttributeError, "IndexError": IndexError,
        "KeyError": KeyError, "NotImplementedError": NotImplementedError,
        "RuntimeError": RuntimeError, "StopIteration": StopIteration,
        "ArithmeticError": ArithmeticError, "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError, "FloatingPointError": FloatingPointError,
        "OSError": OSError, "IOError": IOError, "FileNotFoundError": FileNotFoundError,
        "AssertionError": AssertionError, "NameError": NameError,
        "globals": globals, "locals": locals,
        # File I/O is permitted (image code reads/writes files); note that the
        # ReadOnlyPath guard protects the original input image during execution.
        "open": open,
        # Restricted os shim (no system/popen/exec/spawn). No raw shutil.
        "os": restricted_os,
        "itertools": __import__('itertools'), "re": __import__('re'), "time": __import__('time'),
        "datetime": __import__('datetime'), "math": __import__('math'), "cmath": __import__('cmath'),
        "collections": __import__('collections'), "json": json, "PIL": __import__('PIL'),
        "random": random, "UnidentifiedImageError": UnidentifiedImageError
    }

    sandbox_globals = {
        "__builtins__": allowed_builtins,
        "os": restricted_os,
        "random": random,
        "string": string,
        "math": math,
        "Image": Image,
        "UnidentifiedImageError": UnidentifiedImageError,
        "numpy": numpy,
        "np": numpy,
        "json": json,
        "re": re,
    }

    if cv2:
        sandbox_globals["cv2"] = cv2

    sandbox_globals_always = sandbox_globals
    sandbox_locals = {
        "image_path": input_image_path,
        "temp_output_dir": temp_output_dir
    }

    # Restore previous execution context if exists
    if previous_execution_context:
        sandbox_locals.update(previous_execution_context.get('locals', {}))

        imports_to_recreate = previous_execution_context.get('globals', {})
        for alias, module_name in imports_to_recreate.items():
            try:
                module_obj = importlib.import_module(module_name)
                sandbox_locals[alias] = module_obj
            except ImportError:
                print(f"Warning: Could not re-import module '{module_name}' (as '{alias}')")

    # Code preprocessing
    code_to_execute = align_first_line_to_second(code_to_execute)

    if autopep8:
        try:
            dedented_code = textwrap.dedent(code_to_execute).strip()
            code_to_execute = autopep8.fix_code(dedented_code, options={'aggressive': 2})
        except Exception:
            pass

    # AST transformations for safety and correctness
    if input_image_path and isinstance(input_image_path, str):
        try:
            tree = ast.parse(code_to_execute)
            transformer = ImagePathTransformer(input_image_path)
            new_tree = transformer.visit(tree)
            if transformer.path_was_replaced and hasattr(ast, 'unparse'):
                code_to_execute = ast.unparse(new_tree)
        except Exception as e:
            print(f"Error during image_path AST transformation: {e}")

    # Crop coordinate clamping
    if input_image_path and os.path.isfile(input_image_path):
        try:
            with Image.open(input_image_path) as img_obj:
                actual_image_width, actual_image_height = img_obj.size

            if actual_image_width and actual_image_height:
                tree = ast.parse(code_to_execute)
                crop_transformer = CropCoordinateTransformer(actual_image_width, actual_image_height)
                new_tree = crop_transformer.visit(tree)
                if crop_transformer.coordinates_clamped and hasattr(ast, 'unparse'):
                    code_to_execute = ast.unparse(new_tree)
        except Exception:
            pass

    # OpenCV namespace fixing
    try:
        tree = ast.parse(code_to_execute)
        cv_transformer = OpenCVNamespaceTransformer(correct_prefix="cv2")
        new_tree = cv_transformer.visit(tree)

        if cv_transformer.namespace_updated and hasattr(ast, 'unparse'):
            code_to_execute = ast.unparse(new_tree)
    except Exception:
        pass

    # Execute code
    captured_stdout = io.StringIO()
    processed_paths_list = []
    error_msg = None
    full_print_output = None

    try:
        with contextlib.redirect_stdout(captured_stdout):
            # Replace path prefixes
            original_prefix = '/mnt/data/temp_processed_images/'
            replacement_target_path = temp_output_dir
            if original_prefix.endswith('/') and not temp_output_dir.endswith('/'):
                replacement_target_path = temp_output_dir + '/'

            # Create subdirectories as needed
            quoted_path_pattern = re.compile(r"(['\"])(" + re.escape(original_prefix) + r"[^'\"]*)\1")
            unique_paths_found = set()
            for match in quoted_path_pattern.finditer(code_to_execute):
                unique_paths_found.add(match.group(2))

            for full_path_str in unique_paths_found:
                if full_path_str.startswith(original_prefix):
                    relative_path_suffix = full_path_str[len(original_prefix):].replace(' ', '')
                    if relative_path_suffix:
                        path_directory_part = os.path.dirname(relative_path_suffix)
                        if path_directory_part:
                            target_subdir = os.path.join(temp_output_dir, path_directory_part)
                            os.makedirs(target_subdir, exist_ok=True)

            code_to_execute = code_to_execute.replace(original_prefix, replacement_target_path)
            code_to_execute = re.sub(r':\.\d{1}f}', ':.8f}', code_to_execute)

            exec(code_to_execute, sandbox_globals, sandbox_locals)

        full_print_output = captured_stdout.getvalue().strip()

        # Extract processed paths
        if "processed_path" in sandbox_locals:
            processed_path_from_code = sandbox_locals["processed_path"]
            if isinstance(processed_path_from_code, str) and \
               processed_path_from_code.startswith(temp_output_dir) and \
               os.path.isfile(processed_path_from_code):
                processed_paths_list.append(processed_path_from_code)
            elif isinstance(processed_path_from_code, str) and \
                 processed_path_from_code.startswith(temp_output_dir) and \
                 not os.path.exists(processed_path_from_code):
                error_msg = f"Sandbox {item_id}: 'processed_path' file does not exist: {processed_path_from_code}"

        # Parse paths from print output
        if full_print_output:
            path_pattern = rf"({re.escape(temp_output_dir)}[^\s\'\"]+\.(?:jpg|jpeg|png|bmp|gif|tiff))"
            possible_image_paths = get_image_paths(temp_output_dir)

            for match in re.finditer(path_pattern, full_print_output):
                potential_path = match.group(1)
                if os.path.isfile(potential_path) and potential_path not in processed_paths_list:
                    processed_paths_list.append(potential_path)

            if not processed_paths_list:
                processed_paths_list = possible_image_paths

    except ImportError as e:
        error_msg = f"Sandbox {item_id}: ImportError: {e}"
    except MemoryError as e:
        error_msg = f"Sandbox {item_id}: MemoryError: {e}"
    except SyntaxError as e:
        error_msg = f"Sandbox {item_id}: SyntaxError: {e}"
    except Exception as e:
        error_msg = f"Sandbox {item_id}: Execution failed: {e}"

    return_dict["processed_paths_list"] = processed_paths_list
    return_dict["print_output"] = full_print_output
    return_dict["error"] = error_msg

    # Save execution context
    picklable_variables = {}
    imports_to_persist = {}

    for name, value in sandbox_locals.items():
        if name == "__builtins__" or name in sandbox_globals_always.keys():
            continue
        if isinstance(value, types.ModuleType):
            imports_to_persist[name] = value.__name__
        else:
            # Only persist values that are actually picklable. The previous
            # code assigned first and never triggered the except (assignment
            # does not pickle), so unpicklable objects (PIL images, ndarrays,
            # cv2 handles) leaked into the context and blew up downstream.
            try:
                pickle.dumps(value)
            except Exception:
                continue
            picklable_variables[name] = value

    return_dict["execution_context"] = {
        'globals': imports_to_persist,
        'locals': picklable_variables
    }

    return return_dict


def execute_code_in_sandbox(
    code_to_execute: str,
    input_image_path: Optional[str],
    item_id: str = "N/A",
    temp_output_dir: str = TEMP_PROCESSED_IMAGES_DIR,
    previous_execution_context: Optional[Dict] = None,
    timeout: int = EXEC_TIME_LIMIT
) -> Tuple[List[str], str, Optional[str], Optional[Dict]]:
    """
    Execute Python code in a restricted sandbox with timeout

    Args:
        code_to_execute: Python code string to execute
        input_image_path: Path to input image file
        item_id: Identifier for logging
        temp_output_dir: Directory for output images
        previous_execution_context: Context from previous execution
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (processed_file_paths, print_output, error_message, execution_context)
    """
    if check_dangerous_code(code_to_execute):
        return [], "", f"Sandbox {item_id}: Code contains dangerous operations. Execution denied.", None

    ensure_temp_dir(temp_output_dir)

    with ReadOnlyPath(input_image_path):
        try:
            result_dict = _sandboxed_execution_target(
                code_to_execute,
                input_image_path,
                temp_output_dir,
                item_id,
                previous_execution_context
            )

            processed_paths_list = result_dict.get("processed_paths_list", [])
            full_print_output = result_dict.get("print_output", "")
            error_msg = result_dict.get("error", None)
            current_execution_context = result_dict.get("execution_context", {'globals': {}, 'locals': {}})

        except SandboxTimeoutError:
            error_msg = f"Sandbox {item_id}: Execution timed out after {timeout} seconds."
            print(error_msg)
            processed_paths_list = []
            full_print_output = ""
            current_execution_context = None

    return processed_paths_list, full_print_output, error_msg, current_execution_context
