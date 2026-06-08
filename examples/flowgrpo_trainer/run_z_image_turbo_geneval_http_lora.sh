# Z-Image-Turbo LoRA FlowGRPO with an external GenEval-compatible HTTP reward.
set -x

# Run the reward service on the remaining GPU, then launch this script with
# CUDA_VISIBLE_DEVICES set to the training GPUs (for example: 1,2,3).
WORKSPACE=${WORKSPACE:-/mnt/andy/verl-omni}
export HF_HOME=${HF_HOME:-/mnt/models/hub}
# Keep all local CUDA devices visible to colocated workers; they select devices
# by local rank. Start the reward server on a chosen GPU separately.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}

geneval_train_path=${GENEVAL_TRAIN_PATH:-$WORKSPACE/data/geneval/z_image/train.parquet}
geneval_test_path=${GENEVAL_TEST_PATH:-$WORKSPACE/data/geneval/z_image/test.parquet}

model_name=${Z_IMAGE_TURBO_MODEL_PATH:-/mnt/models/hub/Z-Image-Turbo}
GENEVAL_REWARD_SERVER_URL=${GENEVAL_REWARD_SERVER_URL:?Set GENEVAL_REWARD_SERVER_URL to the scorer endpoint, e.g. http://127.0.0.1:19083}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-3}
ROLLOUT_TP=${ROLLOUT_TP:-1}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-6}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-6}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
SAVE_FREQ=${SAVE_FREQ:-30}
TEST_FREQ=${TEST_FREQ:-30}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}

ENGINE=vllm_omni

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

output_dir=$repo_root/outputs/$script_name
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_images
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=False \
    data.train_files=$geneval_train_path \
    data.val_files=$geneval_test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=512 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora_dtype=fp32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.pipeline.guidance_scale=0.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=512 \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=1 \
    actor_rollout_ref.rollout.algo.sde_window_range="[1,4]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=0.0 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.geneval.path=pkg://verl_omni.utils.reward_score.http_scorer_client" \
    '+reward.reward_functions.geneval.name=compute_score' \
    '+reward.reward_functions.geneval.weight=1.0' \
    "+reward.reward_functions.geneval.server_url=$GENEVAL_REWARD_SERVER_URL" \
    reward.aggregation=weighted_sum \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=z_image_turbo_geneval_http_lora \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    trainer.log_val_generations=$LOG_VAL_GENERATIONS \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS "$@"
