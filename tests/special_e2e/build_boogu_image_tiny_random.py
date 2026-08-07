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
"""Build a tiny Boogu-Image-Turbo checkpoint for smoke tests.

The processor (Qwen3VL tokenizer + chat template) is copied from the source
checkpoint because prompt encoding depends on its special-token IDs and the
chat template; everything else is random-weight and shrunk.

Structural invariants preserved from the real checkpoint:
- ``hidden_size // num_attention_heads == sum(axes_dim_rope)`` (transformer
  config validation),
- ``instruction_feature_configs.instruction_feat_dim`` equals the Qwen3VL
  text hidden size (the transformer consumes ``mllm.model`` last hidden
  states directly),
- VAE ``latent_channels == transformer.in_channels`` (16) and 4 encoder
  blocks so the VAE stride stays 8 (``BOOGU_VAE_SCALE_FACTOR``),
- ``model_index.json`` ``_class_name`` is ``BooguImageTurboPipeline`` (the
  registry dispatches on it).

Usage:
    python tests/special_e2e/build_boogu_image_tiny_random.py \
        --output-dir ~/models/tiny-random/Boogu-Image-Turbo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch
from diffusers import AutoencoderKL
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/models/tiny-random/Boogu-Image-Turbo")
DEFAULT_SOURCE_MODEL = "/mnt/andy/models/Boogu-Image-0.1-Turbo"

# Qwen3VL special-token ids (must match the copied processor/tokenizer).
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653
_BOS_TOKEN_ID = 151643
_EOS_TOKEN_ID = 151645
_VOCAB_SIZE = 151936

# Shared tiny text hidden size == transformer instruction_feat_dim.
_TEXT_HIDDEN_SIZE = 32


def get_dummy_components(*, seed: int = 42) -> dict[str, Any]:
    """Instantiate tiny Boogu pipeline components with random weights."""
    from verl_omni.pipelines.boogu_image_flow_grpo.boogu_model import BooguImageTransformer2DModel

    torch.manual_seed(seed)
    # head_dim = hidden_size / heads = 32 / 2 = 16 == sum(axes_dim_rope).
    transformer = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=32,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=16,
        norm_eps=1e-5,
        axes_dim_rope=(8, 4, 4),
        axes_lens=(512, 64, 64),
        instruction_feature_configs=dict(
            instruction_feat_dim=_TEXT_HIDDEN_SIZE,
            num_instruction_feature_layers=1,
            reduce_type="mean",
        ),
        prompt_tuning_configs=dict(use_prompt_tuning=False),
        timestep_scale=1000.0,
    )

    torch.manual_seed(seed + 1)
    # 4 encoder blocks -> VAE stride 8 (matches BOOGU_VAE_SCALE_FACTOR).
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=(16, 16, 16, 16),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1,
        norm_num_groups=16,
        sample_size=64,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_post_quant_conv=False,
        use_quant_conv=False,
    )

    torch.manual_seed(seed + 2)
    text_num_heads = 2
    text_head_dim = 16
    mllm_config = Qwen3VLConfig(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
        tie_word_embeddings=True,
        text_config=dict(
            vocab_size=_VOCAB_SIZE,
            hidden_size=_TEXT_HIDDEN_SIZE,
            head_dim=text_head_dim,
            num_hidden_layers=2,
            num_attention_heads=text_num_heads,
            num_key_value_heads=1,
            intermediate_size=_TEXT_HIDDEN_SIZE * 2,
            rms_norm_eps=1e-6,
            rope_theta=5000000.0,
            bos_token_id=_BOS_TOKEN_ID,
            eos_token_id=_EOS_TOKEN_ID,
            # mrope_section must sum to head_dim // 2.
            rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [4, 2, 2]},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=_TEXT_HIDDEN_SIZE,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_channels=3,
            num_position_embeddings=64,
            deepstack_visual_indexes=[1],
        ),
    )
    mllm = Qwen3VLForConditionalGeneration(mllm_config)

    return {"transformer": transformer, "vae": vae, "mllm": mllm}


def _copy_pretrained_assets(source_model: str, output_dir: str) -> None:
    """Copy processor and scheduler config files from the source checkpoint.

    Plain file copies: the processor must keep its exact chat template and
    token ids, and the scheduler config is never loaded by the verl-omni
    adapters (they build ``FlowMatchSDEDiscreteScheduler`` directly) but is
    kept for checkpoint-layout fidelity.
    """
    src = os.path.expanduser(source_model)
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Source checkpoint {src!r} not found; pass --source-model pointing to a local "
            "Boogu-Image-0.1-Turbo download (only processor/ and scheduler/ files are read)."
        )
    for subfolder in ("processor", "scheduler"):
        src_dir = os.path.join(src, subfolder)
        dst_dir = os.path.join(output_dir, subfolder)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            if os.path.isfile(src_file) and not fname.endswith((".py", ".bin", ".safetensors")):
                shutil.copy2(src_file, os.path.join(dst_dir, fname))


def _write_model_index(output_dir: str) -> None:
    """Write ``model_index.json`` with the real checkpoint's ``_class_name``."""
    model_index = {
        "_class_name": "BooguImageTurboPipeline",
        "_diffusers_version": "0.35.2",
        "mllm": ["transformers", "Qwen3VLForConditionalGeneration"],
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["scheduling_flow_match_euler_discrete_time_shifting", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["transformer_boogu", "BooguImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)


def build(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Construct and save a tiny random-weight Boogu-Image-Turbo checkpoint."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    components = get_dummy_components(seed=seed)
    components["transformer"].to(dtype).save_pretrained(os.path.join(output_dir, "transformer"))
    components["vae"].to(dtype).save_pretrained(os.path.join(output_dir, "vae"))
    components["mllm"].to(dtype).save_pretrained(os.path.join(output_dir, "mllm"))

    _copy_pretrained_assets(source_model, output_dir)
    _write_model_index(output_dir)
    return output_dir


def ensure_tiny_boogu_image_checkpoint(
    output_dir: str,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    skip_if_exists: bool = True,
) -> str:
    """Build the tiny checkpoint only if it is not already present."""
    output_dir = os.path.expanduser(output_dir)
    if skip_if_exists and os.path.isfile(os.path.join(output_dir, "model_index.json")):
        return output_dir
    return build(output_dir, source_model=source_model, seed=seed, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny Boogu-Image-Turbo checkpoint offline (random weights).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Local checkpoint to copy processor/scheduler files from (small text files only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when output-dir already contains model_index.json",
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if args.force and os.path.isdir(os.path.expanduser(args.output_dir)):
        shutil.rmtree(os.path.expanduser(args.output_dir))
    output_dir = ensure_tiny_boogu_image_checkpoint(
        args.output_dir,
        source_model=args.source_model,
        seed=args.seed,
        dtype=dtype,
        skip_if_exists=not args.force,
    )
    print(f"Tiny Boogu-Image-Turbo checkpoint ready at {output_dir}")


if __name__ == "__main__":
    main()
