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
"""Preprocess GenEval prompts for Z-Image FlowGRPO HTTP-reward training."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _default_split_paths(input_dir: str) -> tuple[str, str]:
    base = Path(os.path.expanduser(input_dir))
    train_path = base / "train.jsonl"
    test_path = base / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train.jsonl and test.jsonl under {base}")
    return str(train_path), str(test_path)


def _make_records(items: list[dict], split: str) -> list[dict]:
    records = []
    for idx, item in enumerate(items):
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            continue
        metadata = {
            "include": item.get("include", []),
            "exclude": item.get("exclude", None),
            "tag": item.get("tag", None),
        }
        records.append(
            {
                "data_source": "geneval",
                "prompt": [{"role": "user", "content": prompt}],
                "negative_prompt": [{"role": "user", "content": " "}],
                "ability": "t2i",
                "reward_model": {"style": "model", "ground_truth": prompt},
                "extra_info": {"split": split, "index": idx, **metadata},
            }
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_dir", default="~/data/geneval", help="Directory containing train/test jsonl.")
    parser.add_argument("--train_path", default=None, help="Explicit train jsonl path.")
    parser.add_argument("--test_path", default=None, help="Explicit test jsonl path.")
    parser.add_argument(
        "--output_dir", default="~/data/geneval/z_image", help="Directory to save the preprocessed parquet files."
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    if args.train_path is None or args.test_path is None:
        train_path, test_path = _default_split_paths(args.input_dir)
    else:
        train_path = os.path.expanduser(args.train_path)
        test_path = os.path.expanduser(args.test_path)

    train_items = _read_jsonl(train_path)
    test_items = _read_jsonl(test_path)
    if args.max_train_samples is not None:
        train_items = train_items[: args.max_train_samples]
    if args.max_test_samples is not None:
        test_items = test_items[: args.max_test_samples]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_records = _make_records(train_items, "train")
    test_records = _make_records(test_items, "test")

    train_output = os.path.join(output_dir, "train.parquet")
    test_output = os.path.join(output_dir, "test.parquet")
    pd.DataFrame(train_records).to_parquet(train_output)
    pd.DataFrame(test_records).to_parquet(test_output)

    print(f"Train: {len(train_records)} records -> {train_output}")
    print(f"Test:  {len(test_records)} records -> {test_output}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
