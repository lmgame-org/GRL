# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

# ─────────────────── MODIFICATION: Import AgentTrainer instead of RayPPOTrainer ───────────────────
from trainer.agent_trainer import AgentTrainer
# ─────────────────── END MODIFICATION ───────────────────
from verl.trainer.ppo.reward import load_reward_manager


# ─────────────────── MODIFICATION: DummyRewardManager replaces load_reward_manager ───────────────────
class DummyRewardManager():
    """The reward manager."""
    
    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score

    def __call__(self, data, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        import torch
        import numpy as np

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            reward_tensor = data.batch['rm_scores']
        else:
            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

            all_scores = []
            already_print_data_sources = {}

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts'] if 'prompts' in data_item.batch else data_item.batch['input_ids']
                prompt_length = prompt_ids.shape[-1] if 'prompts' in data_item.batch else data_item.batch['input_ids'].shape[-1] - data_item.batch['responses'].shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:] if 'prompts' in data_item.batch else data_item.batch['input_ids'][-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() if 'prompts' in data_item.batch else data_item.batch['attention_mask'].sum() - valid_prompt_length
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                # Get score from non_tensor_batch if available
                if hasattr(data_item, 'non_tensor_batch') and data_item.non_tensor_batch is not None:
                    score = data_item.non_tensor_batch.get('reward', 0.0)
                else:
                    score = 0.0
                
                score = float(score)
     
                reward_tensor[i, valid_response_length - 1] = score
                all_scores.append(score)

                # Get data_source from data_item if available, otherwise use a default value
                data_source = data_item.non_tensor_batch.get('data_source', 'unknown') if hasattr(data_item, 'non_tensor_batch') and data_item.non_tensor_batch is not None else 'unknown'
                
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1


        # Handle return_dict parameter
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {}  # Empty dict for now, can be expanded later
            }
        else:
            return reward_tensor
# ─────────────────── END MODIFICATION ───────────────────

# ─────────────────── MODIFICATION: Check config ───────────────────
def check_config(config):
    if len(config.rollout.validation_agent_group_num) != len(config.rollout.validation_agent_group_size):
        raise ValueError("validation_agent_group_num and validation_agent_group_size must have the same length")
    if len(config.rollout.validation) != len(config.rollout.validation_agent_group_num):
        raise ValueError("validation and validation_agent_group_num must have the same length")
    if len(config.rollout.agent_group_num) != len(config.rollout.agent_group_size):
        raise ValueError("agent_group_num and agent_group_size must have the same length")
    if len(config.rollout.training) != len(config.rollout.agent_group_num):
        raise ValueError("training and agent_group_num must have the same length")
# ─────────────────── END MODIFICATION ───────────────────

@hydra.main(config_path="../configs", config_name="ppo_trainer", version_base=None)
def main(config):

    check_config(config)
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # ─────────────────── MODIFICATION: Use DummyRewardManager instead of load_reward_manager ───────────────────
        reward_fn = DummyRewardManager(tokenizer=tokenizer, num_examine=0, compute_score=None)
        val_reward_fn = DummyRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=None)
        # ─────────────────── END MODIFICATION ───────────────────
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # ─────────────────── MODIFICATION: Remove dataset preparation - using None instead of create_rl_dataset ───────────────────
        train_dataset = None
        val_dataset = None
        train_sampler = None
        # ─────────────────── END MODIFICATION ───────────────────

        # ─────────────────── MODIFICATION: Initialize AgentTrainer instead of RayPPOTrainer ───────────────────
        trainer = AgentTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=None,  # Changed from train_dataset
            val_dataset=None,    # Changed from val_dataset
            collate_fn=collate_fn,
            train_sampler=None,  # Changed from train_sampler
            device_name=config.trainer.device,
        )
        # ─────────────────── END MODIFICATION ───────────────────
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()
    

# ─────────────────── MODIFICATION: Remove dataset creation functions - using None datasets instead ───────────────────
# def create_rl_dataset(...) - REMOVED
# def create_rl_sampler(...) - REMOVED
# ─────────────────── END MODIFICATION ───────────────────


if __name__ == "__main__":
    main()
