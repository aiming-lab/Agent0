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
Agent0-VL SFT Dataset

Handles multi-turn reasoning trajectories with:
- Tool calls (JSON parsing)
- Image inputs (vision-language)
- Proper tokenization of <think>...</think> tags
- Support for multiple data sources
"""

import json
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import List, Union, Optional, Callable, Dict, Any

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


def collate_fn(data_list: list[dict]) -> dict:
    """Collate function for DataLoader that handles both tensors and non-tensors."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[dict, Image.Image, str], 
                  max_pixels: int = 2048 * 2048, 
                  min_pixels: int = 512 * 512) -> Image.Image:
    """
    Process image to ensure proper size and format.
    
    Args:
        image: Can be a dict with 'bytes', PIL Image, or file path string
        max_pixels: Maximum number of pixels (will resize if larger)
        min_pixels: Minimum number of pixels (will resize if smaller)
    
    Returns:
        PIL Image in RGB format
    """
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Resize if too large
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    # Resize if too small
    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from response text.
    
    Tool calls are expected in JSON format like:
    {"tool_name": "python_exec", "tool_input": "print(1+1)"}
    
    Returns:
        List of parsed tool call dictionaries
    """
    tool_calls = []
    
    # Find JSON objects in the response
    import re
    json_pattern = r'\{[^{}]*"tool_name"[^{}]*\}'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            if 'tool_name' in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    return tool_calls


class Agent0SFTDataset(Dataset):
    """
    SFT Dataset for Agent0-VL training.
    
    Supports:
    - Multi-turn reasoning trajectories
    - Vision-language inputs (images)
    - Tool call parsing
    - <think>...</think> tag handling
    - Multiple data sources
    
    Expected data format:
    {
        "prompt": [{"role": "user", "content": "..."}],  # or string
        "response": "<think>Step 1: ...</think>\n{\"tool_name\":...}\n...",
        "images": [...],  # PIL images, paths, or bytes
        "data_source": "geometry3k",
    }
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: Union[str, PreTrainedTokenizer],
        processor: Optional[ProcessorMixin] = None,
        prompt_key: str = 'prompt',
        response_key: str = 'response',
        image_key: str = 'images',
        data_source_key: str = 'data_source',
        max_length: int = 8192,
        max_prompt_length: Optional[int] = None,
        truncation: str = 'right',
        filter_overlong: bool = False,
        cache_dir: str = '~/.cache/verl/agent0_sft',
        num_workers: Optional[int] = None,
        # Image processing
        max_pixels: int = 2048 * 2048,
        min_pixels: int = 512 * 512,
        # When True, only tensor fields (input_ids/attention_mask/position_ids/
        # loss_mask) are returned so batches can be wrapped in a TensorDict by
        # trainers that don't handle non-tensor entries (e.g. FSDPSFTTrainer).
        tensor_only: bool = False,
    ):
        """
        Initialize Agent0 SFT Dataset.
        
        Args:
            parquet_files: Path(s) to parquet file(s)
            tokenizer: Tokenizer or path to tokenizer
            processor: Optional processor for vision-language models
            prompt_key: Key for prompt in data
            response_key: Key for response in data
            image_key: Key for images in data
            data_source_key: Key for data source identifier
            max_length: Maximum sequence length (prompt + response)
            max_prompt_length: Maximum prompt length (optional, for filtering)
            truncation: Truncation strategy ('left', 'right', 'error')
            filter_overlong: Whether to filter out overlong sequences
            cache_dir: Directory for caching downloaded files
            num_workers: Number of workers for data processing
            max_pixels: Maximum pixels for image resizing
            min_pixels: Minimum pixels for image resizing
        """
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        
        self.parquet_files = list(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        
        # Setup tokenizer
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        
        # Data keys
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.image_key = image_key
        self.data_source_key = data_source_key
        
        # Length settings
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.filter_overlong = filter_overlong
        
        # Image settings
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.tensor_only = tensor_only
        
        # Workers
        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() // 4)
        else:
            self.num_workers = min(num_workers, os.cpu_count())
        
        self._download()
        self._read_files()

    def _download(self):
        """Download parquet files to local cache if needed."""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(
                src=parquet_file, 
                cache_dir=self.cache_dir,
                verbose=True
            )

    def _read_files(self):
        """Read and concatenate all parquet files."""
        dataframes = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f'Agent0 SFT Dataset loaded: {len(self.dataframe)} samples')
        
        # Optionally filter overlong sequences
        if self.filter_overlong and self.max_prompt_length:
            original_len = len(self.dataframe)
            # Simple length check based on character count (rough estimate)
            self.dataframe = self.dataframe[
                self.dataframe[self.prompt_key].apply(
                    lambda x: len(str(x)) < self.max_prompt_length * 4  # rough char estimate
                )
            ]
            print(f'Filtered {original_len - len(self.dataframe)} overlong samples')

    def __len__(self):
        return len(self.dataframe)

    def _process_prompt(self, prompt) -> List[Dict[str, str]]:
        """Convert prompt to chat format if needed."""
        # Parquet round-trips python lists as numpy arrays
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            return [{'role': 'user', 'content': prompt}]
        elif isinstance(prompt, list):
            return [dict(m) for m in prompt]
        elif isinstance(prompt, dict):
            return [dict(prompt)]
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    def _process_images(self, images) -> List[Image.Image]:
        """Process images from various formats."""
        if images is None:
            return []
        
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        processed = []
        for img in images:
            try:
                processed.append(process_image(img, self.max_pixels, self.min_pixels))
            except Exception as e:
                print(f"Warning: Failed to process image: {e}")
                continue
        
        return processed

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - position_ids: Position IDs (3D for Qwen2-VL)
            - loss_mask: Mask for loss computation (1 for response tokens)
            - data_source: Data source identifier
            - multi_modal_inputs: Image inputs (if applicable)
        """
        row = self.dataframe.iloc[item]
        
        # Get prompt and response
        prompt = row.get(self.prompt_key, '')
        response = row.get(self.response_key, '')
        data_source = row.get(self.data_source_key, 'unknown')
        
        # Process prompt to chat format
        prompt_messages = self._process_prompt(prompt)
        
        # Check for images
        images = row.get(self.image_key, None)
        has_images = images is not None and (
            (isinstance(images, (list, tuple)) and len(images) > 0) or
            (not isinstance(images, (list, tuple)) and images is not None)
        )
        
        # Process images if present
        processed_images = []
        image_grid_thw = None
        multi_modal_inputs = {}
        
        if has_images and self.processor is not None:
            processed_images = self._process_images(images)
            if processed_images:
                # Process images with the processor
                image_inputs = self.processor.image_processor(
                    processed_images, 
                    return_tensors='pt'
                )
                image_grid_thw = image_inputs.get('image_grid_thw', None)
                multi_modal_inputs = {k: v for k, v in image_inputs.items()}
        
        # Apply chat template to prompt (fall back to plain concatenation for
        # tokenizers without a chat template, e.g. in unit tests)
        if getattr(self.tokenizer, 'chat_template', None):
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt_str = "\n".join(str(m.get('content', '')) for m in prompt_messages) + "\n"
        
        # Handle image placeholders for Qwen2-VL
        if has_images and self.processor is not None and image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size ** 2
            index = 0
            while '<image>' in prompt_str:
                num_tokens = image_grid_thw[index].prod() // merge_length
                prompt_str = prompt_str.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * num_tokens + '<|vision_end|>',
                    1
                )
                index += 1
            prompt_str = prompt_str.replace('<|placeholder|>', self.processor.image_token)
        
        # Prepare response with EOS token
        response_str = response + self.tokenizer.eos_token
        
        # Tokenize prompt
        prompt_output = self.tokenizer(
            prompt_str, 
            return_tensors='pt', 
            add_special_tokens=False
        )
        prompt_ids = prompt_output['input_ids'][0]
        prompt_attention_mask = prompt_output['attention_mask'][0]
        prompt_length = prompt_ids.shape[0]
        
        # Tokenize response
        response_output = self.tokenizer(
            response_str, 
            return_tensors='pt', 
            add_special_tokens=False
        )
        response_ids = response_output['input_ids'][0]
        response_attention_mask = response_output['attention_mask'][0]
        response_length = response_ids.shape[0]
        
        # Concatenate prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        
        # Handle sequence length
        sequence_length = input_ids.shape[0]
        
        if sequence_length < self.max_length:
            # Pad to max_length
            pad_length = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id or 0
            
            padded_input_ids = torch.full(
                (pad_length,), 
                pad_token_id, 
                dtype=input_ids.dtype
            )
            padded_attention_mask = torch.zeros(pad_length, dtype=attention_mask.dtype)
            
            input_ids = torch.cat([input_ids, padded_input_ids])
            attention_mask = torch.cat([attention_mask, padded_attention_mask])
            
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                # Adjust prompt_length for loss mask
                prompt_length = max(0, prompt_length - (sequence_length - self.max_length))
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise ValueError(
                    f'Sequence length {sequence_length} exceeds max_length {self.max_length}'
                )
        
        # Create loss mask (1 for response tokens, 0 for prompt and padding)
        loss_mask = attention_mask.clone()
        
        # Mask out prompt tokens (keep only response for loss). Loss is computed
        # on shifted logits, so mask index j gates the prediction of token j+1;
        # keep index (prompt_length-1) so the first response token is supervised.
        if prompt_length > 0:
            loss_mask[:min(max(0, prompt_length - 1), loss_mask.size(0))] = 0
        
        # Mask out the last token (EOS) - we don't predict after EOS
        actual_length = min(prompt_length + response_length, loss_mask.size(0))
        if actual_length > 0:
            loss_mask[actual_length - 1] = 0
        
        # Compute position IDs
        if has_images and self.processor is not None and image_grid_thw is not None:
            # Use Qwen2-VL specific position IDs (3D for mrope)
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            # Standard position IDs
            position_ids = compute_position_id_with_mask(attention_mask)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }

        if self.tensor_only:
            return result

        result['data_source'] = data_source
        # Add multi-modal inputs if present
        if multi_modal_inputs:
            result['multi_modal_inputs'] = multi_modal_inputs
            result['multi_modal_data'] = {'image': processed_images}

        return result


class Agent0MultiTurnSFTDataset(Dataset):
    """
    Multi-turn SFT Dataset for Agent0-VL training.
    
    Handles conversations with multiple turns where each assistant response
    should be trained. Supports tool calls and verification steps.
    
    Expected data format:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<think>...</think>\n{tool_call}..."},
            {"role": "user", "content": "Tool output: ..."},
            {"role": "assistant", "content": "FINAL_ANSWER: ..."},
        ],
        "images": [...],
        "data_source": "geometry3k",
    }
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: Union[str, PreTrainedTokenizer],
        processor: Optional[ProcessorMixin] = None,
        messages_key: str = 'messages',
        image_key: str = 'images',
        data_source_key: str = 'data_source',
        max_length: int = 16384,
        truncation: str = 'right',
        cache_dir: str = '~/.cache/verl/agent0_sft',
        # Image processing
        max_pixels: int = 2048 * 2048,
        min_pixels: int = 512 * 512,
        # Loss mask options
        train_on_all_assistant: bool = True,  # Train on all assistant turns
    ):
        """
        Initialize Agent0 Multi-turn SFT Dataset.
        
        Args:
            parquet_files: Path(s) to parquet file(s)
            tokenizer: Tokenizer or path to tokenizer
            processor: Optional processor for vision-language models
            messages_key: Key for messages list in data
            image_key: Key for images in data
            data_source_key: Key for data source identifier
            max_length: Maximum sequence length
            truncation: Truncation strategy
            cache_dir: Directory for caching
            max_pixels: Maximum pixels for image resizing
            min_pixels: Minimum pixels for image resizing
            train_on_all_assistant: Whether to train on all assistant turns
        """
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        
        self.parquet_files = list(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        
        self.messages_key = messages_key
        self.image_key = image_key
        self.data_source_key = data_source_key
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.train_on_all_assistant = train_on_all_assistant
        
        self._download()
        self._read_files()

    def _download(self):
        """Download parquet files to local cache if needed."""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(
                src=parquet_file,
                cache_dir=self.cache_dir,
                verbose=True
            )

    def _read_files(self):
        """Read and concatenate all parquet files."""
        dataframes = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f'Agent0 Multi-turn SFT Dataset loaded: {len(self.dataframe)} samples')

    def __len__(self):
        return len(self.dataframe)

    def _process_images(self, images) -> List[Image.Image]:
        """Process images from various formats."""
        if images is None:
            return []
        
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        processed = []
        for img in images:
            try:
                processed.append(process_image(img, self.max_pixels, self.min_pixels))
            except Exception as e:
                print(f"Warning: Failed to process image: {e}")
                continue
        
        return processed

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a single multi-turn training sample."""
        row = self.dataframe.iloc[item]
        
        messages = row.get(self.messages_key, [])
        data_source = row.get(self.data_source_key, 'unknown')
        images = row.get(self.image_key, None)
        
        # Process images
        has_images = images is not None and (
            (isinstance(images, (list, tuple)) and len(images) > 0) or
            (not isinstance(images, (list, tuple)))
        )
        
        processed_images = []
        image_grid_thw = None
        multi_modal_inputs = {}
        
        if has_images and self.processor is not None:
            processed_images = self._process_images(images)
            if processed_images:
                image_inputs = self.processor.image_processor(
                    processed_images,
                    return_tensors='pt'
                )
                image_grid_thw = image_inputs.get('image_grid_thw', None)
                multi_modal_inputs = {k: v for k, v in image_inputs.items()}
        
        # Tokenize full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Handle image placeholders for Qwen2-VL
        if has_images and self.processor is not None and image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size ** 2
            index = 0
            while '<image>' in full_text:
                num_tokens = image_grid_thw[index].prod() // merge_length
                full_text = full_text.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * num_tokens + '<|vision_end|>',
                    1
                )
                index += 1
            full_text = full_text.replace('<|placeholder|>', self.processor.image_token)
        
        # Tokenize
        full_output = self.tokenizer(
            full_text,
            return_tensors='pt',
            add_special_tokens=False
        )
        input_ids = full_output['input_ids'][0]
        attention_mask = torch.ones_like(input_ids)
        
        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        
        # Find positions of assistant responses
        if self.train_on_all_assistant:
            # Train on all assistant turns
            current_pos = 0
            for i, msg in enumerate(messages):
                # Tokenize up to this message
                prefix_messages = messages[:i+1]
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Handle image placeholders in prefix
                if has_images and self.processor is not None and image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size ** 2
                    idx = 0
                    while '<image>' in prefix_text:
                        num_tokens = image_grid_thw[idx].prod() // merge_length
                        prefix_text = prefix_text.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * num_tokens + '<|vision_end|>',
                            1
                        )
                        idx += 1
                    prefix_text = prefix_text.replace('<|placeholder|>', self.processor.image_token)
                
                prefix_output = self.tokenizer(
                    prefix_text,
                    return_tensors='pt',
                    add_special_tokens=False
                )
                end_pos = prefix_output['input_ids'].shape[1]
                
                if msg['role'] == 'assistant':
                    # Set loss mask for this assistant turn
                    loss_mask[current_pos:end_pos] = 1
                
                current_pos = end_pos
        else:
            # Only train on the last assistant turn
            if messages and messages[-1]['role'] == 'assistant':
                prefix_messages = messages[:-1]
                if prefix_messages:
                    prefix_text = self.tokenizer.apply_chat_template(
                        prefix_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # Handle image placeholders
                    if has_images and self.processor is not None and image_grid_thw is not None:
                        merge_length = self.processor.image_processor.merge_size ** 2
                        idx = 0
                        while '<image>' in prefix_text:
                            num_tokens = image_grid_thw[idx].prod() // merge_length
                            prefix_text = prefix_text.replace(
                                '<image>',
                                '<|vision_start|>' + '<|placeholder|>' * num_tokens + '<|vision_end|>',
                                1
                            )
                            idx += 1
                        prefix_text = prefix_text.replace('<|placeholder|>', self.processor.image_token)
                    
                    prefix_output = self.tokenizer(
                        prefix_text,
                        return_tensors='pt',
                        add_special_tokens=False
                    )
                    start_pos = prefix_output['input_ids'].shape[1]
                else:
                    start_pos = 0
                
                loss_mask[start_pos:] = 1
        
        # Handle sequence length
        sequence_length = input_ids.shape[0]
        
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id or 0
            
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=attention_mask.dtype)
            ])
            loss_mask = torch.cat([
                loss_mask,
                torch.zeros(pad_length, dtype=loss_mask.dtype)
            ])
            
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise ValueError(
                    f'Sequence length {sequence_length} exceeds max_length {self.max_length}'
                )
        
        # Compute position IDs
        if has_images and self.processor is not None and image_grid_thw is not None:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'data_source': data_source,
        }
        
        if multi_modal_inputs:
            result['multi_modal_inputs'] = multi_modal_inputs
            result['multi_modal_data'] = {'image': processed_images}
        
        return result
