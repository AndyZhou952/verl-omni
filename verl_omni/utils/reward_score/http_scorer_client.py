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

"""Generic HTTP reward client for external scorer services.

Sends generated images to an external HTTP scorer service using pickle protocol
and returns the score. Compatible with all scorer services under
rewards_services/api_services/ that accept the standard payload format::

    POST with pickle-serialized {"images": List[bytes], "prompts": List[str], "metadata": dict}
    Response: pickle-serialized {"scores": List[float]}
"""

import asyncio
import io
import json
import logging
import pickle

import aiohttp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to a uint8 RGB PIL image."""
    if image.ndim == 4:
        image = image[0]
    image = image.float().permute(1, 2, 0).cpu().numpy()
    image = (image * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _serialize_image(pil_image: Image.Image) -> bytes:
    """Serialize a PIL image to JPEG bytes."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return buf.getvalue()


def _decode_metadata_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _build_scorer_metadata(extra_info: dict) -> dict:
    return {
        key: _decode_metadata_value(value)
        for key, value in extra_info.items()
        if key not in {"num_turns", "rollout_reward_scores", "split", "index"}
    }


def _prepare_image_bytes(image: torch.Tensor) -> bytes:
    """Convert image tensor to JPEG bytes (CPU-heavy, run in thread pool)."""
    pil_image = _tensor_to_pil(image)
    return _serialize_image(pil_image)


async def compute_score(
    solution_image: torch.Tensor,
    ground_truth: str,
    server_url: str,
    extra_info: dict | None = None,
    metadata: dict | None = None,
    request_format: str = "generic",
    only_strict: bool = False,
    fail_on_error: bool = False,
    **kwargs,
) -> dict:
    """Compute reward by calling an external HTTP scorer service.

    Args:
        solution_image: Generated image tensor (C, H, W) or (N, C, H, W).
        ground_truth: Prompt string passed directly to the scorer service.
        server_url: Full URL of the scorer service (e.g., "http://localhost:19082").
        extra_info: Per-sample metadata from the dataset. GenEval-style services
            can read include/exclude/tag fields from the forwarded metadata.
        metadata: Explicit metadata override. If omitted, metadata is derived
            from extra_info while dropping reward-loop bookkeeping fields.
        request_format: Payload variant. Use "geneval" for yifan123/reward-server's
            GenEval service, which expects meta_datas and only_strict.
        only_strict: Forwarded to GenEval services when request_format="geneval".
        fail_on_error: Raise on HTTP/service errors instead of returning 0.

    Returns:
        dict with "score" key.
    """
    loop = asyncio.get_event_loop()
    image_bytes = await loop.run_in_executor(None, _prepare_image_bytes, solution_image)
    if metadata is None:
        extra_info = extra_info or {}
        metadata = _build_scorer_metadata(extra_info)
    if request_format == "geneval":
        metadata = dict(metadata)
        metadata.setdefault("prompt", ground_truth)

    payload_dict = {
        "images": [image_bytes],
        "prompts": [ground_truth],
        "metadata": metadata,
    }
    if request_format == "geneval":
        payload_dict.update(
            {
                "meta_datas": [metadata],
                "only_strict": only_strict,
            }
        )
    payload = pickle.dumps(payload_dict)

    if not hasattr(compute_score, "_session") or compute_score._session.closed:
        timeout = aiohttp.ClientTimeout(total=120)
        compute_score._session = aiohttp.ClientSession(timeout=timeout)

    session = compute_score._session
    async with session.post(server_url, data=payload) as resp:
        if resp.status != 200:
            error_text = await resp.text()
            message = f"Scorer server returned {resp.status}: {error_text}"
            logger.error(message)
            if fail_on_error:
                raise RuntimeError(message)
            return {"score": 0.0}
        response_data = pickle.loads(await resp.read())

    if "error" in response_data:
        message = f"Scorer server error: {response_data['error']}"
        logger.error(message)
        if fail_on_error:
            raise RuntimeError(message)
        return {"score": 0.0}

    scores = response_data["scores"]
    score = float(scores[0]) if scores else 0.0
    result = {"score": score}
    for key in ("rewards", "strict_rewards"):
        values = response_data.get(key)
        if values:
            result[key.rstrip("s")] = float(values[0])
    for key in ("group_rewards", "group_strict_rewards"):
        group_values = response_data.get(key, {})
        if isinstance(group_values, dict):
            for group_name, values in group_values.items():
                if values:
                    result[f"{key.rstrip('s')}/{group_name}"] = float(values[0])
    return result
