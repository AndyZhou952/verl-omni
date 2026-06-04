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
"""Preprocess DanceGRPO prompts for Z-Image HPSv3 FlowGRPO training."""

import argparse
import os
import random
import re

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return [prompt for prompt in prompts if not _contains_chinese(prompt)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--input_path",
        default="~/data/hpsv3/video_prompts.txt",
        help="Path to the raw DanceGRPO video_prompts.txt file.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/hpsv3/z_image",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Fraction of prompts reserved for the test set.",
    )

    args = parser.parse_args()
    input_path = os.path.expanduser(args.input_path)
    output_dir = os.path.expanduser(args.output_dir)

    prompts = _load_prompts(input_path)
    print(f"Loaded {len(prompts)} prompts (after filtering Chinese lines)")

    data_source = "flow_grpo/hpsv3"
    negative_user_prompt = " "

    def make_record(prompt: str, split: str, idx: int) -> dict:
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "negative_prompt": [{"role": "user", "content": negative_user_prompt}],
            "ability": "t2i",
            "reward_model": {"style": "model", "ground_truth": prompt},
            "extra_info": {"split": split, "index": idx},
        }

    random.seed(42)
    random.shuffle(prompts)
    test_count = max(1, min(int(len(prompts) * args.test_ratio), 1000))
    test_prompts = prompts[:test_count]
    train_prompts = prompts[test_count:]

    train_records = [make_record(prompt, "train", idx) for idx, prompt in enumerate(train_prompts)]
    test_records = [make_record(prompt, "test", idx) for idx, prompt in enumerate(test_prompts)]

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    pd.DataFrame(train_records).to_parquet(train_path)
    pd.DataFrame(test_records).to_parquet(test_path)

    print(f"Train: {len(train_records)} records -> {train_path}")
    print(f"Test:  {len(test_records)} records -> {test_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
