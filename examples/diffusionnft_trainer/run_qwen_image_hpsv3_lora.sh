# Qwen-Image DiffusionNFT LoRA RL with HPSv3 reward, vllm_omni rollout
set -x

# Set WORKSPACE to the directory containing data/.
WORKSPACE=${WORKSPACE:-/mnt/andy/verl-omni}
export HF_HOME=${HF_HOME:-/mnt/models/hub}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}

hpsv3_train_path=${HPSV3_TRAIN_PATH:-$WORKSPACE/data/hpsv3/qwen_image_nft/train.parquet}
hpsv3_test_path=${HPSV3_TEST_PATH:-$WORKSPACE/data/hpsv3/qwen_image_nft/test.parquet}

model_name=${QWEN_IMAGE_MODEL_PATH:-/mnt/models/hub/Qwen-Image}
tokenizer_path=${QWEN_IMAGE_TOKENIZER_PATH:-$model_name/tokenizer}
reward_model_name=${HPSV3_REWARD_MODEL_PATH:-/mnt/models/hub/HPSv3/HPSv3.safetensors}
reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py
export custom_reward_model_path=$reward_model_name

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
REWARD_DEVICE=${REWARD_DEVICE:-cuda}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-256}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4}
TRUE_CFG_SCALE=${TRUE_CFG_SCALE:-4.0}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MICRO_BATCH_SIZE=$((MICRO_BATCH_SIZE_PER_GPU * NUM_GPUS_ACTOR_ROLLOUT_REWARD))
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$MICRO_BATCH_SIZE}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((MINI_BATCH_SIZE * N_RESP_PER_PROMPT))}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}

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
    data.train_files=$hpsv3_train_path \
    data.val_files=$hpsv3_test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_path \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.5 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=$TRUE_CFG_SCALE \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    algorithm.adv_mode=continuous \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.reward_model.model_path=$reward_model_name \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.hpsv3.path=$repo_root/$reward_function_path" \
    '+reward.reward_functions.hpsv3.name=compute_score_hpsv3' \
    '+reward.reward_functions.hpsv3.weight=1.0' \
    "+reward.reward_functions.hpsv3.device=$REWARD_DEVICE" \
    reward.aggregation=weighted_sum \
    trainer.logger='["console"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=qwen_image_hpsv3_lora \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS "$@"
