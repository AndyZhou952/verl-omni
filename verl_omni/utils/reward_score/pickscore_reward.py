# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""PickScore reward for text-to-image RL.

This mirrors FlowFactory's PickScore reward recipe:
`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` processor,
`yuvalkirstain/PickScore_v1` model, and raw score divided by 26.
"""

import threading
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.utils.generic import ModelOutput

DEFAULT_PROCESSOR_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEFAULT_MODEL_PATH = "yuvalkirstain/PickScore_v1"

_lock = threading.Lock()
_models: dict[tuple[str, str, str, str], tuple[CLIPProcessor, CLIPModel]] = {}


def _extract_feature_tensor(output: Any) -> torch.Tensor:
    """Handle transformers versions that return either tensors or ModelOutput."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, ModelOutput):
        return output.pooler_output
    raise TypeError(f"Unexpected feature output type: {type(output).__name__}")


def _to_pil_hwc(image: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
    """Normalize tensor/array/PIL image to uint8 RGB PIL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().float()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def _extract_images(solution_image, frame_interval: int = 1) -> list[Image.Image]:
    """Extract one or more PIL images from image/video tensors."""
    if isinstance(solution_image, Image.Image):
        return [solution_image.convert("RGB")]

    if isinstance(solution_image, np.ndarray):
        solution_image = torch.from_numpy(solution_image)

    if not isinstance(solution_image, torch.Tensor):
        return [_to_pil_hwc(solution_image)]

    is_channels_last = solution_image.ndim >= 3 and solution_image.shape[-1] in (1, 3)

    if solution_image.ndim == 3:
        return [_to_pil_hwc(solution_image)]

    if solution_image.ndim == 4:
        if is_channels_last:
            frames = solution_image[::frame_interval]
        elif solution_image.shape[0] in (1, 3):
            frames = solution_image[:, ::frame_interval].permute(1, 0, 2, 3)
        else:
            frames = solution_image[::frame_interval]
        return [_to_pil_hwc(frame) for frame in frames]

    if solution_image.ndim == 5:
        if is_channels_last:
            solution_image = solution_image.permute(0, 4, 1, 2, 3)
        frames = solution_image[:, :, ::frame_interval].permute(0, 2, 1, 3, 4)
        frames = frames.reshape(-1, *frames.shape[2:])
        return [_to_pil_hwc(frame) for frame in frames]

    raise ValueError(f"Unsupported image tensor shape: {tuple(solution_image.shape)}")


def _torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_model(
    model_path: str,
    processor_path: str,
    device: str,
    dtype: str | torch.dtype | None,
) -> tuple[CLIPProcessor, CLIPModel]:
    dtype_obj = _torch_dtype(dtype)
    dtype_key = str(dtype_obj) if dtype_obj is not None else "auto"
    key = (model_path, processor_path, device, dtype_key)
    if key not in _models:
        processor = CLIPProcessor.from_pretrained(processor_path)
        model = CLIPModel.from_pretrained(model_path, torch_dtype=dtype_obj)
        model.eval().to(device)
        _models[key] = (processor, model)
    return _models[key]


@torch.no_grad()
def compute_score_pickscore(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    model_path: str = DEFAULT_MODEL_PATH,
    processor_path: str = DEFAULT_PROCESSOR_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
    score_scale: float = 26.0,
    **kwargs,
) -> dict:
    """Compute normalized PickScore for a generated image and prompt."""
    prompt = ground_truth if ground_truth else ""
    frame_interval = extra_info.get("frame_interval", 1)
    images = _extract_images(solution_image, frame_interval=frame_interval)

    with _lock:
        processor, model = _get_model(model_path, processor_path, device, dtype)

        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {key: value.to(device=device) for key, value in image_inputs.items()}

        text_inputs = processor(
            text=[prompt] * len(images),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device=device) for key, value in text_inputs.items()}

        image_embs = _extract_feature_tensor(model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = _extract_feature_tensor(model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        raw_scores = model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)

    raw_score = raw_scores.float().mean().item()
    return {"score": raw_score / score_scale, "pickscore_raw": raw_score}
