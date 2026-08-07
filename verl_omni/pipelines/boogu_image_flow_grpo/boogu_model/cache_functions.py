# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Vendored from https://github.com/boogu-project/Boogu-Image (Apache-2.0),
# boogu/cache_functions/cal_type.py and boogu/cache_functions/force_scheduler.py, commit a366095.
# Modified: merged cal_type.py and force_scheduler.py into a single module
#           (cache_init.py omitted; it is pipeline-level and not needed by cal_type).
# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: TaylorSeer (Shenyi-Z), taylorseer_flux/cache_functions/cal_type.py
# and taylorseer_flux/cache_functions/force_scheduler.py
# Source: https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/

import torch


def force_scheduler(cache_dic, current):
    """
    Update `cache_dic['cal_threshold']` for the current denoising step.

    Args:
        cache_dic: Mutable cache state dict. Expected keys include
            `fresh_ratio` and `fresh_threshold`.
        current: Per-step state dict. Expected keys include
            `step` and `num_steps`.
    """
    if cache_dic["fresh_ratio"] == 0:
        # FORA
        linear_step_weight = 0.0
    else:
        # TokenCache
        linear_step_weight = 0.0
    # Scale threshold by step position when linear weighting is enabled.
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current["step"] / current["num_steps"])
    threshold = torch.round(cache_dic["fresh_threshold"] / step_factor)

    # no force constrain for sensitive steps, cause the performance is good enough.
    # you may have a try.

    cache_dic["cal_threshold"] = threshold


def cal_type(cache_dic, current):
    """
    Determine the compute mode for the current step.

    Side effects:
        - Updates `current['type']` to one of: 'full', 'Taylor', 'ToCa', 'Delta-Cache'.
        - Updates `cache_dic['cache_counter']`.
        - Updates scheduling threshold via `force_scheduler` on full-refresh steps.
    """
    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA:Uniform
        first_step = current["step"] == 0
    else:
        # ToCa: First enhanced
        first_step = current["step"] < cache_dic["first_enhance"]

    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        # Full compute refresh: reset counter and update adaptive threshold.
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        force_scheduler(cache_dic, current)

    elif cache_dic["taylor_cache"]:
        # Reuse with Taylor approximation between full-refresh steps.
        cache_dic["cache_counter"] += 1
        current["type"] = "Taylor"

    elif cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    # 'cache_noise' 'ToCa' 'FORA'
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
