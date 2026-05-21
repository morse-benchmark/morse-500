# Configured for single-GPU A5000 (24 GB VRAM).
#
# Memory strategy — two models must share 24 GB VRAM but NEVER simultaneously:
#   Rollout phase : vLLM weights (~9 GB) on GPU; actor FSDP offloaded to CPU
#   Training phase: actor FSDP on GPU (~9 GB weights + ~9 GB grads ≈ 20 GB peak);
#                   vLLM sleeps with sleep_level=2 (frees ALL GPU memory, vLLM>=0.8.5)
#
# Key settings:
#   - actor.fsdp_config.param_offload=True: actor weights on CPU between rollout/training;
#     VERL manually loads/offloads via load_fsdp_model_to_gpu / offload_fsdp_model_to_cpu
#   - rollout.free_cache_engine=True: vLLM sleeps (level=2) after rollout, freeing GPU
#     entirely before the actor training step (requires vLLM>=0.8.5; container has 0.17.0)
#   - gpu_memory_utilization=0.85: vLLM alone on 24 GB; 0.85×24=20.4 GB target;
#     with ~9 GB weights, ~11 GB left for KV cache (ample for 2048-token seqs at n=4)
#   - optimizer_offload=True + AdamW8bit: optimizer on CPU (~9 GB quantized moments)
#   - ref.fsdp_config.param_offload=True: ref model always on CPU
#   - ray start --head --object-store-memory=8GB: caps Ray object store (default ~72 GB)
#   - +ray_kwargs.ray_init.address=auto: connects VERL to the pre-started Ray cluster

module load gcc/11.2.0
module load cuda/12.6.3
unset ROCR_VISIBLE_DEVICES

# Use local node /tmp for Ray/TMPDIR so sessions don't consume the shared NFS scratch disk
export RAY_TMPDIR=/tmp/ray_${USER}
export TMPDIR=/tmp/verl_${USER}
# Suppress Ray's verbose internal logging (heartbeats, state dumps, dashboard noise)
export RAY_LOG_TO_STDERR=0
export RAY_DEDUP_LOGS=1
export VLLM_LOGGING_LEVEL=WARNING
export RAY_LOGGING_LEVEL=WARNING
export PYTHONWARNINGS=ignore
# Point HF cache to the existing hf_cache on NFS.
# The apptainer command must bind this path: --bind ...,/fs/nexus-scratch/mrislam/hf_cache
export HF_HOME=/fs/nexus-scratch/mrislam/hf_cache
mkdir -p $RAY_TMPDIR $TMPDIR

# Start Ray manually so object_store_memory is enforced.
# When VERL calls ray.init() it connects to this running instance instead of
# spawning a new one with Ray's default (~30% of system RAM = ~67 GB).
ray stop --force 2>/dev/null || true
ray start --head \
    --temp-dir=$RAY_TMPDIR \
    --object-store-memory=8589934592 \
    --num-cpus=8 \
    --num-gpus=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/fs/nexus-scratch/mrislam/morse-500/fine_tuning/data/train_morse_grpo.parquet \
    data.val_files=/fs/nexus-scratch/mrislam/morse-500/fine_tuning/data/val_morse_grpo.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.video_key=video \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-4B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.optimizer=AdamW8bit \
    actor_rollout_ref.actor.optim.optimizer_impl=bitsandbytes.optim \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='morse500_grpo' \
    trainer.experiment_name='qwen3_4b_morse' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    +ray_kwargs.ray_init.address=auto \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
