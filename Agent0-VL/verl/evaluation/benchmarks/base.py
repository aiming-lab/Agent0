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
Benchmark loading for Agent0-VL evaluation.

Each benchmark resolves data in this order:
1. A local file inside ``data_dir``: ``{name}.parquet``, ``{name}.json`` or
   ``{name}.jsonl`` (columns: question / answer or ground_truth / optional
   image or image_path).
2. The HuggingFace dataset configured on the benchmark class (downloaded via
   ``datasets.load_dataset``).

Loaders raise on failure — evaluation must never silently produce an empty
benchmark. Normalized samples contain: ``id``, ``question``, ``ground_truth``,
``data_source`` and optionally ``image_pil`` (PIL image) / ``image_path``.

NOTE: for publication-grade numbers use an external harness such as VLMEvalKit
(https://github.com/open-compass/VLMEvalKit); this lightweight loader is meant
for quick tracking of model quality during development.
"""

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _to_pil(image) -> Optional[Any]:
    """Best-effort conversion of dataset image fields to a PIL RGB image."""
    if image is None:
        return None
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict) and "bytes" in image and image["bytes"]:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, str) and os.path.exists(image):
            with Image.open(image) as img:
                return img.convert("RGB").copy()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to decode image field: {e}")
    return None


class BaseBenchmark:
    """Generic benchmark loader (local file first, HuggingFace fallback)."""

    # Subclasses configure these
    name: str = ""
    hf_repo: str = ""
    hf_config: Optional[str] = None
    hf_split: str = "test"
    # Field names in the raw dataset
    question_field: str = "question"
    answer_field: str = "answer"
    image_field: str = "image"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    def load_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = self._load_local()
        source = "local"
        if rows is None:
            rows = self._load_hf()
            source = "huggingface"

        samples = []
        for idx, row in enumerate(rows):
            if max_samples is not None and idx >= max_samples:
                break
            sample = self.normalize_sample(row, idx)
            if sample is not None:
                samples.append(sample)

        if not samples:
            raise RuntimeError(
                f"Benchmark '{self.name}' produced 0 samples from {source} source. "
                f"Place a file at {self.data_dir}/{self.name}.parquet (columns: "
                f"question, answer, optional image/image_path) or check access to "
                f"HuggingFace dataset '{self.hf_repo}'."
            )
        logger.info(f"Loaded {len(samples)} samples for benchmark '{self.name}' ({source})")
        return samples

    # ------------------------------------------------------------------
    def _local_candidates(self) -> List[str]:
        return [
            os.path.join(self.data_dir, f"{self.name}.parquet"),
            os.path.join(self.data_dir, f"{self.name}.json"),
            os.path.join(self.data_dir, f"{self.name}.jsonl"),
            os.path.join(self.data_dir, self.name, "test.parquet"),
        ]

    def _load_local(self) -> Optional[List[Dict[str, Any]]]:
        for path in self._local_candidates():
            if not os.path.exists(path):
                continue
            logger.info(f"Loading benchmark '{self.name}' from local file {path}")
            if path.endswith(".parquet"):
                import pandas as pd

                return pd.read_parquet(path).to_dict("records")
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("data", [])
        return None

    def _load_hf(self) -> List[Dict[str, Any]]:
        if not self.hf_repo:
            raise RuntimeError(f"Benchmark '{self.name}' has no local data and no HF repo configured")
        from datasets import load_dataset

        logger.info(f"Downloading benchmark '{self.name}' from HuggingFace: "
                    f"{self.hf_repo} (config={self.hf_config}, split={self.hf_split})")
        if self.hf_config:
            ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        else:
            ds = load_dataset(self.hf_repo, split=self.hf_split)
        return ds

    # ------------------------------------------------------------------
    def normalize_sample(self, row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = row.get(self.question_field) or row.get("question") or row.get("prompt") or ""
        answer = (row.get(self.answer_field) if row.get(self.answer_field) is not None
                  else row.get("answer", row.get("ground_truth", "")))
        if isinstance(answer, (list, tuple)):
            answer = answer[0] if len(answer) > 0 else ""

        if not question:
            return None

        sample: Dict[str, Any] = {
            "id": row.get("id", row.get("sample_id", f"{self.name}_{idx}")),
            "question": str(question),
            "ground_truth": str(answer),
            "data_source": self.name,
        }

        image = row.get(self.image_field) or row.get("image") or row.get("decoded_image")
        pil = _to_pil(image)
        if pil is not None:
            sample["image_pil"] = pil
        elif isinstance(row.get("image_path"), str):
            sample["image_path"] = row["image_path"]
        return sample


class MathVerseBenchmark(BaseBenchmark):
    name = "mathverse"
    hf_repo = "AI4Math/MathVerse"
    hf_config = "testmini"
    hf_split = "testmini"
    image_field = "image"


class MathVistaBenchmark(BaseBenchmark):
    name = "mathvista"
    hf_repo = "AI4Math/MathVista"
    hf_split = "testmini"
    image_field = "decoded_image"


class MathVisionBenchmark(BaseBenchmark):
    name = "mathvision"
    hf_repo = "MathLLMs/MathVision"
    hf_split = "test"
    image_field = "decoded_image"


class WeMathBenchmark(BaseBenchmark):
    name = "wemath"
    hf_repo = "We-Math/We-Math"
    hf_split = "testmini"
    image_field = "image_path"


class ChartQABenchmark(BaseBenchmark):
    name = "chartqa"
    hf_repo = "HuggingFaceM4/ChartQA"
    hf_split = "test"
    question_field = "query"
    answer_field = "label"
    image_field = "image"


class HallBenchBenchmark(BaseBenchmark):
    name = "hallbench"
    hf_repo = "lmms-lab/HallusionBench"
    hf_split = "image"
    answer_field = "gt_answer"
    image_field = "image"


class MMMUBenchmark(BaseBenchmark):
    name = "mmmu"
    hf_repo = "lmms-lab/MMMU"
    hf_split = "validation"
    image_field = "image_1"

    def normalize_sample(self, row, idx):
        sample = super().normalize_sample(row, idx)
        if sample is not None and row.get("options"):
            options = row["options"]
            if isinstance(options, str):
                try:
                    import ast

                    options = ast.literal_eval(options)
                except (ValueError, SyntaxError):
                    options = [options]
            letters = "ABCDEFGH"
            option_lines = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))
            sample["question"] = f"{sample['question']}\nOptions:\n{option_lines}"
        return sample


BENCHMARK_REGISTRY = {
    "mathverse": MathVerseBenchmark,
    "mathvista": MathVistaBenchmark,
    "mathvision": MathVisionBenchmark,
    "wemath": WeMathBenchmark,
    "chartqa": ChartQABenchmark,
    "hallbench": HallBenchBenchmark,
    "mmmu": MMMUBenchmark,
}


def get_benchmark(name: str, data_dir: str) -> BaseBenchmark:
    """Instantiate a benchmark loader by name (case-insensitive)."""
    key = name.lower()
    if key not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARK_REGISTRY.keys())}")
    return BENCHMARK_REGISTRY[key](data_dir)
