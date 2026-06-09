# Z-Image-Turbo LoRA RL with PickScore reward, vllm_omni rollout
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

pickscore_train_path=$WORKSPACE/data/pickscore/z_image/train.parquet
pickscore_test_path=$WORKSPACE/data/pickscore/z_image/test.parquet

model_name=Tongyi-MAI/Z-Image-Turbo
pickscore_model_path=yuvalkirstain/PickScore_v1
pickscore_processor_path=laion/CLIP-ViT-H-14-laion2B-s32B-b79K
reward_function_path=verl_omni/utils/reward_score/pickscore_reward.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=1
IMAGE_RESOLUTION=512

ENGINE=vllm_omni

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=False \
    data.train_files=$pickscore_train_path \
    data.val_files=$pickscore_test_path \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora_dtype=fp32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=6 \
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
    "+reward.reward_functions.pickscore.path=$reward_function_path" \
    '+reward.reward_functions.pickscore.name=compute_score_pickscore' \
    '+reward.reward_functions.pickscore.weight=1.0' \
    "+reward.reward_functions.pickscore.model_path=$pickscore_model_path" \
    "+reward.reward_functions.pickscore.processor_path=$pickscore_processor_path" \
    '+reward.reward_functions.pickscore.device=cuda' \
    '+reward.reward_functions.pickscore.dtype=bfloat16' \
    reward.aggregation=weighted_sum \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=z_image_turbo_pickscore_t2is_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
