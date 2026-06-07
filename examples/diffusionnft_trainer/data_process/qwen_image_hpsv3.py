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
"""Convert HPSv3 text prompts to Qwen-Image DiffusionNFT parquet format."""

import argparse
import os

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)


def _prompt_text(row: pd.Series) -> str:
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth")
    if ground_truth:
        return ground_truth

    prompt = row.get("prompt")
    if prompt is not None:
        for message in prompt:
            if message.get("role") == "user":
                return message.get("content", "")
    return ""


def _normalize_row(row: pd.Series) -> pd.Series:
    text = _prompt_text(row)
    row = row.copy()
    row["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    row["negative_prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": " "},
    ]
    row["ability"] = row.get("ability") or "t2i"
    row["reward_model"] = {"style": "model", "ground_truth": text}
    return row


def convert_split(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)
    out = df.apply(_normalize_row, axis=1)
    out.to_parquet(output_path)
    print(f"Wrote {len(out)} records -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--input_dir",
        default="~/data/hpsv3/z_image",
        help="Directory containing HPSv3 train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/hpsv3/qwen_image_nft",
        help="Directory to save Qwen-Image formatted HPSv3 parquet files.",
    )
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    convert_split(os.path.join(input_dir, "train.parquet"), os.path.join(output_dir, "train.parquet"))
    convert_split(os.path.join(input_dir, "test.parquet"), os.path.join(output_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
