# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Vendored from https://github.com/boogu-project/Boogu-Image (Apache-2.0),
# boogu/models/transformers/components.py, commit a366095.
import torch.nn.functional as F


def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y
