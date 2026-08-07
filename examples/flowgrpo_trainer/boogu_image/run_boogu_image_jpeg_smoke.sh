# Boogu-Image-0.1-Turbo FlowGRPO smoke run: rule reward only (JPEG
# compressibility), no reward model, few training steps.
#
# Validates the full loop (rollout -> rule reward -> flow_grpo -> full-weight
# FSDP update -> weight sync back to rollout) without a GenRM reward server.
#
# FULL-WEIGHT ONLY (no LoRA): the self-contained Boogu rollout pipeline uses
# plain nn.Linear layers, so LoRA weight sync is not supported.
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/boogu_image/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/boogu_image/test.parquet

model_name=${MODEL_PATH:-$WORKSPACE/models/Boogu-Image-0.1-Turbo}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-4}
ROLLOUT_TP=1
IMAGE_RESOLUTION=512

ENGINE=vllm_omni

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name/processor \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,3]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.project_name=flow_grpo \
    trainer.experiment_name=boogu_image_jpeg_smoke \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=5 "$@"
