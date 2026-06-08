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
"""Preprocess T2IS/PickScore prompts for Z-Image FlowGRPO training."""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def _read_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("text")
            else:
                prompt = line
            if prompt:
                prompts.append(prompt)
    return prompts


def _default_split_paths(input_dir: str) -> tuple[str, str | None]:
    base = Path(os.path.expanduser(input_dir))
    for suffix in ("jsonl", "txt"):
        train_path = base / f"train.{suffix}"
        test_path = base / f"test.{suffix}"
        if train_path.exists():
            return str(train_path), str(test_path) if test_path.exists() else None
    raise FileNotFoundError(f"Could not find train.jsonl or train.txt under {base}")


def _make_records(prompts: list[str], split: str, data_source: str) -> list[dict]:
    return [
        {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "negative_prompt": [{"role": "user", "content": " "}],
            "ability": "t2i",
            "reward_model": {"style": "model", "ground_truth": prompt},
            "extra_info": {"split": split, "index": idx},
        }
        for idx, prompt in enumerate(prompts)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_dir", default="~/data/t2is", help="Directory containing train/test jsonl or txt.")
    parser.add_argument("--train_path", default=None, help="Explicit train jsonl/txt path.")
    parser.add_argument("--test_path", default=None, help="Explicit test jsonl/txt path.")
    parser.add_argument(
        "--output_dir", default="~/data/t2is/z_image", help="Directory to save the preprocessed parquet files."
    )
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test split ratio if no test file exists.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_path is None:
        train_path, inferred_test_path = _default_split_paths(args.input_dir)
        test_path = args.test_path or inferred_test_path
    else:
        train_path = os.path.expanduser(args.train_path)
        test_path = os.path.expanduser(args.test_path) if args.test_path else None

    train_prompts = _read_prompts(train_path)
    if test_path is not None:
        test_prompts = _read_prompts(test_path)
    else:
        random.seed(args.seed)
        random.shuffle(train_prompts)
        test_count = max(1, min(int(len(train_prompts) * args.test_ratio), 1000))
        test_prompts = train_prompts[:test_count]
        train_prompts = train_prompts[test_count:]

    if args.max_train_samples is not None:
        train_prompts = train_prompts[: args.max_train_samples]
    if args.max_test_samples is not None:
        test_prompts = test_prompts[: args.max_test_samples]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_records = _make_records(train_prompts, "train", "flow_factory/t2is")
    test_records = _make_records(test_prompts, "test", "flow_factory/t2is")

    train_output = os.path.join(output_dir, "train.parquet")
    test_output = os.path.join(output_dir, "test.parquet")
    pd.DataFrame(train_records).to_parquet(train_output)
    pd.DataFrame(test_records).to_parquet(test_output)

    print(f"Train: {len(train_records)} records -> {train_output}")
    print(f"Test:  {len(test_records)} records -> {test_output}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
