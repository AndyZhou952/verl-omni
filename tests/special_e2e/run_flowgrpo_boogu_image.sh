#!/usr/bin/env bash
# FlowGRPO Boogu-Image-Turbo e2e smoke test (minimal runtime), vllm_omni rollout.
#
# Single pass covering:
#   parquet load -> self-contained vllm_omni rollout pipeline (type B) ->
#   jpeg_compressibility rule reward -> flow_grpo -> FSDP FULL-WEIGHT update ->
#   full-weight sync back to rollout via load_weights ("transformer." prefix).
#
# FULL-WEIGHT ONLY: the Boogu rollout pipeline is built from plain nn.Linear
# layers, so LoRA weight sync is not supported (lora_rank stays 0).
#
# Requires: vllm-omni, diffusers>=0.37,
#   tiny Boogu-Image-Turbo at ~/models/tiny-random/Boogu-Image-Turbo
#   (built automatically when missing; the builder copies processor files
#   from BOOGU_SOURCE_MODEL, default /mnt/andy/models/Boogu-Image-0.1-Turbo)
set -euo pipefail

# Override via env: NUM_GPUS, MODEL_PATH, BOOGU_SOURCE_MODEL, DATA_DIR,
#                   TOTAL_TRAIN_STEPS, TRAIN_FILES, VAL_FILES
NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Boogu-Image-Turbo}
BOOGU_SOURCE_MODEL=${BOOGU_SOURCE_MODEL:-/mnt/andy/models/Boogu-Image-0.1-Turbo}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/processor}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

ENGINE=vllm_omni
max_prompt_length=256

# Build the tiny-random checkpoint on demand (cheap, offline).
if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_boogu_image_tiny_random.py \
        --output-dir "${MODEL_PATH}" \
        --source-model "${BOOGU_SOURCE_MODEL}"
fi

# This helper runs nvidia-smi in a background loop during training and
# fails if any vLLMOmniHttpServer process appears.
_LEAK_FILE=$(mktemp)
_LEAK_PID=""
cleanup_leak_monitor() {
    [[ -n "${_LEAK_PID}" ]] && kill "${_LEAK_PID}" 2>/dev/null || true
    rm -f "${_LEAK_FILE}"
}
trap cleanup_leak_monitor EXIT

start_leak_monitor() {
    : > "${_LEAK_FILE}"
    while true; do
        if nvidia-smi -i 0 2>&1 | grep -q "vLLMOmniHttpServer"; then
            echo "LEAK" >> "${_LEAK_FILE}"
        fi
        sleep 1
    done &
    _LEAK_PID=$!
}

check_leak_monitor() {
    kill "${_LEAK_PID}" 2>/dev/null || true
    _LEAK_PID=""
    if grep -q "LEAK" "${_LEAK_FILE}" 2>/dev/null; then
        echo ""
        echo "FAIL: unexpected vLLMOmniHttpServer process(es) detected on GPU-0 —"
        ray stop --force 2>/dev/null || true
        exit 1
    fi
}

n_resp_per_prompt=2
micro_bsz_per_gpu=1
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4

start_leak_monitor
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=64 \
    actor_rollout_ref.rollout.pipeline.width=64 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,3]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.jpeg.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility" \
    '+reward.reward_functions.jpeg.name=compute_score' \
    '+reward.reward_functions.jpeg.weight=1.0' \
    reward.aggregation=weighted_sum \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-boogu-image-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"
check_leak_monitor

echo "FlowGRPO Boogu-Image e2e test passed (training completed successfully)."
