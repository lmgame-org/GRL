# sync_multi_turn_rollout.py
from typing import List, Dict, Any, Union
import torch
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from lmgamerl.agents import get_agent_cls, REGISTERED_AGENTS
import numpy as np
from tensordict import TensorDict


class SyncMultiTurnRollout:
    """
    Vectorised, synchronous, multi-turn rollout manager.
    Batch size = cfg.agent_length (fallback to cfg.env_length).
    
    Each agent manages its own environment, history, and recorder internally.
    This class orchestrates the batch processing and LLM calls.
    """

    # ─────────────────── INITIALIZATION ───────────────────
    def __init__(self, actor_rollout_wg, cfg, tokenizer, processor, validation=False):
        """
        Initialize rollout manager. Agent class is resolved from config.
        
        Args:
            actor_rollout_wg: Actor rollout worker group
            cfg: Configuration object
            tokenizer: Tokenizer for text processing
            processor: Data processor
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_wg = actor_rollout_wg
        
        # Calculate total agents from agent_group_num * agent_group_size
        if validation:
            self.agent_group_num_list = getattr(cfg.rollout, "validation_agent_group_num", [64])
            self.agent_group_size_list = getattr(cfg.rollout, "validation_agent_group_size", [1])
        else:
            self.agent_group_num_list = getattr(cfg.rollout, "agent_group_num", [4])
            self.agent_group_size_list = getattr(cfg.rollout, "agent_group_size", [2])
            
        self.n_agents_list = [agent_group_num * agent_group_size for agent_group_num, agent_group_size in zip(self.agent_group_num_list, self.agent_group_size_list)]
        self.total_group_num = sum(self.agent_group_num_list)
        self.total_agent_num = sum(self.n_agents_list)
        self.validation = validation



        # Initialize agent configuration from config
        self._setup_agent_config()
        self._init_batch_agents()
        
        # Global turn counter
        self.step_cnt = 0

    def _setup_agent_config(self):
        """
        Setup agent configuration for single agent type.
        Agent class is resolved from config and agent config is extracted.
        """
        # Get agent name from rollout config train list (first item)
        if self.validation:
            self.agent_names = getattr(self.cfg.rollout, 'validation', ['simpleSokobanAgent'])
        else:
            self.agent_names = getattr(self.cfg.rollout, 'training', ['simpleSokobanAgent'])
        
        self.agent_cls_list = []
        self.agent_config_list = []
        self.max_turns_list = []
        for agent_name in self.agent_names:
            agent_type = self.cfg[agent_name]['agent_type']

            # Resolve agent class from registry
            self.agent_cls_list.append(get_agent_cls(agent_type))

            self.agent_config_list.append(self.cfg[agent_name])
            self.max_turns_list.append(self.cfg[agent_name]['agent_config'].get('max_turns', 5))
        
        
        # Get max_turns from agent config
        self.max_turns = max(self.max_turns_list)

    def _init_batch_agents(self):
        """
        Build self.agents: List[Agent] without resetting them.
        Each agent handles its own history & recorder.
        Agents are grouped based on agent_group_size for training purposes.
        Actual reset happens in rollout() via _reset_batch_agents().
        """
        # Create agents of the same type
        if len(self.agent_cls_list) == 0:
            raise ValueError("agent_cls_list is None but trying to create agents")
        
        # Verify the math
        for i, agent_num in enumerate(self.n_agents_list):
            if agent_num != self.agent_group_num_list[i] * self.agent_group_size_list[i]:
                raise ValueError(f"Total agents ({agent_num}) != agent_group_num ({self.agent_group_num_list[i]}) × agent_group_size ({self.agent_group_size_list[i]})")

        self.agents = []
        done_groups = 0
        agent_id_counter = 0

        # loop through all agent types
        for i, agent_cls in enumerate(self.agent_cls_list):
            group_num = self.agent_group_num_list[i]
            group_size = self.agent_group_size_list[i]
            cfg = self.agent_config_list[i]
            name = self.agent_names[i]
            # loop through all groups of the same agent type
            for local_group_id in range(group_num):
                global_group_id = local_group_id + done_groups
                # loop through group_size to initialize agents
                for _ in range(group_size):
                    agent = agent_cls(
                        config=cfg,
                        agent_id=agent_id_counter,
                        group_id=global_group_id,
                        tag=name)
                    agent_id_counter += 1
                    self.agents.append(agent)
            # update the done_groups for the next agent type
            done_groups += self.agent_group_num_list[i]
        
        # Initialize tracking structures - actual env_outs will be set in rollout()
        self.done_mask = torch.zeros(self.total_agent_num, dtype=torch.bool)
        self.env_outs = None  # Will be initialized in _reset_batch_agents()

    # ─────────────────── BATCH LLM PROMPTS ───────────────────
    def get_batch_llm_prompts(self, env_outputs):
        """
        Generate batch of LLM prompts from environment outputs.
        Each agent.get_llm_prompts(env_out) returns messages format.
        
        Args:
            env_outputs: List of environment outputs from agents
            
        Returns:
            DataProto: Batched DataProto containing input_ids, attention_mask, position_ids
        """
        llm_input_texts = []
        
        for idx, env_out in enumerate(env_outputs):
            if self.done_mask[idx]:
                # For done agents, use empty prompt
                llm_input_texts.append("")
                continue
                
            agent = self.agents[idx]
            
            # Each agent returns messages format
            messages = agent.get_llm_prompts(env_out)
            
            # Apply chat template to convert messages to text
            try:
                prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                prompt_str = "System error in chat template"

            if agent.agent_config.get('use_think_answer_token', True):
                if agent.agent_config.get('enable_think', True):
                    prompt_str += "<think>"
                else:
                    prompt_str += "<answer>"
            
            llm_input_texts.append(prompt_str)
        
        # Tokenize all prompts using verl_F for more universal processing
        batch_list = []
        
        for i, prompt_str in enumerate(llm_input_texts):
            
            if not prompt_str or len(prompt_str.strip()) == 0:
                prompt_str = "Please respond."  # Fallback
            
            # Use verl_F.tokenize_and_postprocess_data for consistent processing
            try:
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt_str,
                    tokenizer=self.tokenizer,
                    max_length=self.cfg.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,  # Left pad for batch generation
                    truncation=self.cfg.rollout.truncation
                )
                
                # Check for all-padding sequences
                pad_token_id = self.tokenizer.pad_token_id
                non_pad_count = (input_ids != pad_token_id).sum().item()
                total_tokens = input_ids.numel()
                
            except Exception as e:
                # Create emergency fallback tokens
                fallback_text = "Please respond."
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=fallback_text,
                    tokenizer=self.tokenizer,
                    max_length=self.cfg.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.cfg.rollout.truncation
                )
            
            # Compute position ids
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            # Build row_dict for each prompt
            row_dict = {
                'input_ids': input_ids.squeeze(0),  # Remove batch dimension
                'attention_mask': attention_mask.squeeze(0),
                'position_ids': position_ids.squeeze(0),
                'responses': input_ids.squeeze(0)[1:],  # Remove first token for responses
            }
            batch_list.append(row_dict)
        
        # Use collate_fn to batch the data, then convert to DataProto
        try:
            batch_dict = collate_fn(batch_list)
            result_dataproto = DataProto.from_single_dict(batch_dict)
            return result_dataproto
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    # ─────────────────── BATCH ENV OUTPUTS ───────────────────
    def get_batch_env_outputs(self, lm_outputs):
        """
        Process LLM outputs and update environment outputs for all agents.
        
        Args:
            lm_outputs: DataProto containing LLM responses
            
        Returns:
            List: Updated environment outputs from all agents
        """
        # Ensure env_outs is initialized (should be done by _reset_batch_agents)
        if self.env_outs is None:
            raise RuntimeError("env_outs not initialized. Call rollout() or _reset_batch_agents() first.")
        
        # Decode responses
        replies = self.tokenizer.batch_decode(
            lm_outputs.batch["responses"], 
            skip_special_tokens=True
        )

        # Update environment outputs for all agents
        updated_env_outs = []
        
        for idx, reply in enumerate(replies):
            if self.done_mask[idx]:
                # Keep existing env output for done agents
                updated_env_outs.append(self.env_outs[idx])
                continue
                
            agent = self.agents[idx]
            # Agent handles history updates internally
            env_out = agent.get_env_outputs(reply)
            updated_env_outs.append(env_out)
            
            # Update tracking structures
            self.env_outs[idx] = env_out
            self.done_mask[idx] = env_out.truncated or env_out.terminated
        
        return updated_env_outs

    # ─────────────────── LLM GENERATION ───────────────────
    def generate_sequences(self, lm_inputs: DataProto):
        """
        Generate sequences using the actor worker group.
        
        Args:
            lm_inputs: DataProto containing input_ids, attention_mask, position_ids
            
        Returns:
            DataProto: Generated sequences
        """
        # TODO: add kv cache both for the vllm wrapper here and for verl vllm.
        try:
            from verl.trainer.ppo.ray_trainer import RayWorkerGroup
            from verl.utils.dataset.rl_dataset import pad_dataproto_to_divisor, unpad_dataproto
        except ImportError:
            RayWorkerGroup = None
            pad_dataproto_to_divisor = None
            unpad_dataproto = None
        
        if (RayWorkerGroup is not None and isinstance(self.actor_wg, RayWorkerGroup) and 
            pad_dataproto_to_divisor is not None and unpad_dataproto is not None):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        else:
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        
        return lm_outputs

    # ─────────────────── MAIN ROLLOUT LOOP ───────────────────
    def rollout(self):
        """
        Main rollout loop using batch LLM prompts approach.
        Iterate cfg.agent.max_turn turns, breaking early if all done.
        """
        self._reset_batch_agents()
        
        for turn in range(self.max_turns):
            if self.done_mask.all():
                break

            # Generate batch of LLM prompts from current env outputs
            batch_prompts = self.get_batch_llm_prompts(self.env_outs)
            
            # Generate responses using batch dispatch
            lm_outputs = self.generate_sequences(batch_prompts)
            
            # Process LLM outputs and update environment outputs
            self.env_outs = self.get_batch_env_outputs(lm_outputs)

            
            self.step_cnt += 1
        

    # ─────────────────── PPO BATCH BUILDING ───────────────────

    def get_masks_and_scores(self, input_ids: torch.Tensor, all_scores: List[List[float]] | None = None, use_turn_scores: bool = False):
        """
        Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
        NOTE: This assumes that the input_ids starts with system and then user & assistant in alternative ways
        
        Args:
            input_ids: shape (bsz, seq_len)
            all_scores: List of score lists for each agent
            use_turn_scores: Whether to use turn-based scores
            
        Returns:
            Tuple of (loss_mask, score_tensor, response_mask)
        """
        special_token = self.tokenizer.encode("<|im_start|>")[0]
        turn_starts = torch.where(input_ids == special_token, 1, 0)
        turn_indicators = torch.cumsum(turn_starts, dim=-1)
        response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
        loss_mask = (turn_indicators > 1) # learns everything after system prompt

        reward_token = self.tokenizer.encode("<|im_end|>")[0]
        score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
        if all_scores is not None:
            if use_turn_scores:
                for idx, scores in enumerate(list(zip(*all_scores))):
                    scores = torch.tensor(scores, dtype=torch.float32)
                    turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
                    reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
                    score_tensor[reward_position] = scores
            else:
                scores = [sum(i) for i in all_scores]
                score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
        loss_mask = loss_mask[:, :-1] # remove the last token
        score_tensor = score_tensor[:, 1:] # remove the first token

        return loss_mask, score_tensor, response_mask

    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.cfg.rollout.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.cfg.rollout.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")

        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # Apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        for group, index in group2index.items():
            normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        # Apply penalty
        penalty = torch.tensor([env_output["penalty"] for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor

    def filter_rollout(self, rollout_batch: DataProto) -> DataProto:
        """
        Filter rollout batch based on the filter ratio.
        """
        rollout_filter_ratio = self.cfg.rollout.rollout_filter_ratio

        # ad hoc set the agent_group_num and agent_group_size, assuming only one type of agent in training
        num_groups, group_size = self.agent_group_num_list[0], self.agent_group_size_list[0]
        
        rm_scores = rollout_batch.batch["rm_scores"].sum(dim=-1).view(num_groups, group_size)
        
        selected_groups = int(rollout_filter_ratio * num_groups)

        in_group_std = rm_scores.std(dim=-1)
        in_group_max = rm_scores.max(dim=-1).values
        in_group_mean = rm_scores.mean(dim=-1)
        if rollout_filter_ratio == 1:
            return rollout_batch, {"rollout/in_group_std": in_group_std.mean(), "rollout/in_group_max": in_group_max.mean(), "rollout/in_group_mean": in_group_mean.mean(), "rollout/chosen_in_group_std": in_group_std.mean(), "rollout/chosen_in_group_max": in_group_max.mean(), "rollout/chosen_in_group_mean": in_group_mean.mean()}

        if self.cfg.rollout.rollout_filter_type == "std_rev":
            top_groups = (-in_group_std).topk(int(rollout_filter_ratio * num_groups)).indices
        elif self.cfg.rollout.rollout_filter_type == "std":
            top_groups = in_group_std.topk(int(rollout_filter_ratio * num_groups)).indices
        else:
            raise ValueError(f"Invalid rollout filter type: {self.cfg.rollout.rollout_filter_type}")
        
        mask = torch.zeros(num_groups, dtype=torch.bool)
        mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

        rollout_batch.batch = rollout_batch.batch[mask]
        
        for key, value in rollout_batch.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                rollout_batch.non_tensor_batch[key] = value[mask]
            else:
                rollout_batch.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]

        metrics = {
            "rollout/in_group_std": in_group_std.mean(),
            "rollout/in_group_max": in_group_max.mean(),
            "rollout/in_group_mean": in_group_mean.mean(),
            "rollout/chosen_in_group_std": in_group_std[top_groups].mean(),
            "rollout/chosen_in_group_max": in_group_max[top_groups].mean(),
            "rollout/chosen_in_group_mean": in_group_mean[top_groups].mean()
        }
        return rollout_batch, metrics

    def _collect_final_rollout_states(self) -> List[Dict]:
        """
        Collect final rollout states from all agents.
        
        Returns:
            List[Dict]: List of rollout state dictionaries from all agents
        """
        env_outputs = []
        for idx in range(self.total_agent_num):
            agent = self.agents[idx]
            rollout_state = agent.get_final_rollout_states()
            env_outputs.append(rollout_state)
        return env_outputs

    def build_ppo_batch(self, rollout_states: List[Dict]) -> DataProto:
        """
        Build PPO batch from the final batch rollout states.
        Converts collected rollout states to DataProto format for PPO training.
        """
        llm_input_texts = []
        messages_list = []
        
        # Loop through all agents to collect their LLM prompts
        for idx, agent in enumerate(self.agents):
            # Get the current environment output for this agent
            env_out = self.env_outs[idx] if self.env_outs else None
            
            if env_out is None:
                # Handle case where env_outs is not initialized
                llm_input_texts.append("")
                continue
            
            # Get messages from agent's get_messages method
            messages = agent.get_messages()
   

            # NOTE: this assertion is important for loss mask computation
            assert all(msg["role"] == "assistant" for msg in messages[2::2])
            
            messages_list.append(messages)
            
            # Apply chat template to convert messages to text
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            except Exception as e:
                # Fallback in case of chat template error
                prompt_text = "System error in chat template"
            
            llm_input_texts.append(prompt_text)
        
        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        scores = [[i['reward'] for i in env_output['history']] for env_output in rollout_states]
        
        loss_mask, score_tensor, response_mask = self.get_masks_and_scores(input_ids, scores, use_turn_scores=self.cfg.rollout.use_turn_scores)
        normalized_score_tensor = self._normalize_score_tensor(score_tensor, rollout_states)
        response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
            'loss_mask': loss_mask,
            'rm_scores': normalized_score_tensor,
        }, batch_size=input_ids.shape[0])

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in rollout_states], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in rollout_states], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        metrics = {}
        n_agents_map = dict(zip(self.agent_names, self.n_agents_list))
        for env_output in rollout_states:
            for key, value in env_output["metrics"].items():
                if key not in metrics:
                        metrics[key] = []
                metrics[key].append(value)
        metrics = {
            key: np.sum(value) / n_agents_map[key.split("/")[0]]
            for key, value in metrics.items()
        }
        metrics["response_length"] = response_length
        llm_inputs.meta_info = {"metrics": metrics}

        return llm_inputs

    # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────

    def _reset_batch_agents(self, seed=None):
        """
        Reset all agents and collect the batch of initial env outputs.
        This function resets the rollout manager for new epoch/rollout.
        
        Args:
            seed: Optional base seed for reproducibility. If None, generates random seed.
        """
        import random
        
        # Generate base seed following reference implementation pattern
        if seed is not None:
            base_seed = seed
        elif self.validation:
            base_seed = self.cfg.rollout.validation_seed
        else:# Generate random seed for training, consistent seed for validation  
            base_seed = random.randint(0, 1000000)
            
        # Generate group seeds: agents within same group share environment
        # Different groups get different environments
        group_seeds = [base_seed + group_id for group_id in range(self.total_group_num)]
        
        initial_env_outs = []
        
        for idx in range(self.total_agent_num):
            agent = self.agents[idx]
            
            # All agents in the same group use the same seed (same environment)
            group_seed = group_seeds[agent.group_id]
            
            # Reset agent with group-specific seed
            initial_env_out = agent.reset(seed=group_seed)
            initial_env_outs.append(initial_env_out)
        
        # Update tracking structures with batch of reset outputs
        self.done_mask = torch.zeros(self.total_agent_num, dtype=torch.bool)
        self.env_outs = initial_env_outs
        self.step_cnt = 0
        
    def reset(self, seed=None):
        """
        Public reset method for external use (e.g., called by trainer between epochs).
        Delegates to _reset_batch_agents() for actual reset logic.
        
        Args:
            seed: Optional base seed for reproducibility
        """
        self._reset_batch_agents(seed=seed)

    def close(self):
        """
        Clean up agents and environments for tidy teardown.
        """
        for idx in range(self.total_agent_num):
            agent = self.agents[idx]
            if hasattr(agent, 'close'):
                agent.close()
            # If agent has separate env reference
            if hasattr(agent, 'env') and hasattr(agent.env, 'close'):
                agent.env.close()
