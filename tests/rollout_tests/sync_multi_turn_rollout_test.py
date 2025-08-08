#!/usr/bin/env python3
"""
SyncMultiTurnRollout Test - Tests rollout logic with mocked LLM responses
Enhanced with debugging for prompt generation issues
"""

import sys
import os
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import real components
from transformers import AutoTokenizer
from lmgamerl.rollout.sync_multi_turn_rollout import SyncMultiTurnRollout
from tests.rollout_tests.rollout_test_utils import create_mock_actor_wg, create_mock_tokenizer_decode


def setup_logging():
    """Setup logging to stream outputs to test_logs directory"""
    test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(test_logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_logs_dir, f"sync_multi_turn_rollout_debug_{timestamp}.log")
    
    class Tee:
        def __init__(self, file_path):
            self.file = open(file_path, 'w')
            self.stdout = sys.stdout
        
        def write(self, data):
            self.file.write(data)
            self.file.flush()
            self.stdout.write(data)
        
        def flush(self):
            self.file.flush()
            self.stdout.flush()
        
        def close(self):
            self.file.close()
    
    tee = Tee(log_file)
    sys.stdout = tee
    
    print(f"📝 SyncMultiTurnRollout DEBUG Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 70)
    
    return tee


def load_config():
    """Load configuration from base.yaml and agents.yaml"""
    config_dir = project_root / "configs"
    
    try:
        # Load base config
        with open(config_dir / "base.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Load agents config
        with open(config_dir / "agents.yaml", 'r') as f:
            agents_config = yaml.safe_load(f)
        
        # Merge configs
        config = {**base_config, **agents_config}
        
        print(f"✅ Loaded configuration from {config_dir}")
        print(f"   Base config keys: {list(base_config.keys())}")
        print(f"   Agents config keys: {list(agents_config.keys())}")
        
        return config
        
    except FileNotFoundError as e:
        print(f"❌ Could not find config file: {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        raise


def create_config_object(config_dict):
    """Create a config object with proper attribute access"""
    class Config:
        def __init__(self, config_dict):
            self._original_dict = config_dict
            for key, value in config_dict.items():
                # Only set attributes for string keys (valid Python identifiers)
                if isinstance(key, str) and key.isidentifier():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        def get(self, key, default=None):
            if hasattr(self, key):
                return getattr(self, key)
            return self._original_dict.get(key, default)
        
        def __contains__(self, key):
            return hasattr(self, key) or key in self._original_dict
        
        def __getitem__(self, key):
            if hasattr(self, key):
                attr_value = getattr(self, key)
                # If it's a Config object, return its _original_dict for agent consumption
                if isinstance(attr_value, Config):
                    return attr_value._original_dict
                return attr_value
            return self._original_dict[key]
        
        def __getattr__(self, key):
            if key.startswith('_'):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            return self._original_dict.get(key, None)
    
    return Config(config_dict)


def create_real_tokenizer():
    """Create a real tokenizer using Hugging Face"""
    try:
        print("📦 Loading Qwen/Qwen2.5-0.5B tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply mock tokenizer decode for testing
        tokenizer = create_mock_tokenizer_decode(tokenizer)
        
        print(f"✅ Real tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load real tokenizer: {e}")
        raise


def debug_tokenization(tokenizer, text, label=""):
    """Debug helper to analyze tokenization results"""
    print(f"\n🔍 TOKENIZATION DEBUG - {label}")
    print("=" * 50)
    
    if not text or text.strip() == "":
        print("❌ EMPTY OR WHITESPACE-ONLY TEXT!")
        print(f"   Raw text: '{text}'")
        print(f"   Length: {len(text)}")
        return None, None
    
    print(f"Input text (first 200 chars): '{text[:200]}{'...' if len(text) > 200 else ''}'")
    print(f"Text length: {len(text)}")
    
    try:
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        
        print(f"✅ Tokenization successful")
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        print(f"   Input IDs: {input_ids[0][:20].tolist()}{'...' if input_ids.shape[1] > 20 else ''}")
        
        # Check for padding-only sequences
        non_pad_count = (input_ids != tokenizer.pad_token_id).sum().item()
        total_tokens = input_ids.numel()
        
        print(f"   Non-padding tokens: {non_pad_count}/{total_tokens}")
        print(f"   Padding ratio: {(total_tokens - non_pad_count)/total_tokens:.2%}")
        
        if non_pad_count == 0:
            print("❌ ALL TOKENS ARE PADDING! This will cause vLLM IndexError!")
            
        # Check attention mask
        valid_attention = attention_mask.sum().item()
        print(f"   Valid attention positions: {valid_attention}/{attention_mask.numel()}")
        
        return input_ids, attention_mask
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return None, None


def ensure_rollout_initialized(rollout):
    """Helper to ensure rollout env_outs is initialized"""
    if rollout.env_outs is None:
        print("   🔄 Initializing rollout environment outputs...")
        rollout._reset_batch_agents()


def test_prompt_generation_debug():
    """Debug test for prompt generation issues"""
    print("🔍 DEBUGGING PROMPT GENERATION...")
    print("=" * 70)
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    print(f"\n📊 INITIAL ROLLOUT STATE:")
    print(f"   Number of agents: {rollout.n_agents}")
    print(f"   Agent group num: {rollout.agent_group_num}")
    print(f"   Agent group size: {rollout.agent_group_size}")
    print(f"   Max turns: {rollout.max_turns}")
    print(f"   Current step: {rollout.step_cnt}")
    
    # Initialize environment outputs
    ensure_rollout_initialized(rollout)
    
    # Check initial environment outputs
    print(f"\n📊 INITIAL ENVIRONMENT OUTPUTS:")
    if rollout.env_outs is not None and len(rollout.env_outs) > 0:
        for i, env_out in enumerate(rollout.env_outs[:3]):  # Show first 3
            print(f"\n   Agent {i} Initial Env Output:")
            print(f"      Done: {env_out.done}")
            print(f"      Reward: {env_out.reward}")
            print(f"      State (first 100 chars): '{env_out.state[:100]}{'...' if len(env_out.state) > 100 else ''}'")
            print(f"      State length: {len(env_out.state)}")
            print(f"      Info keys: {list(env_out.info.keys())}")
    else:
        print(f"   ❌ No environment outputs available - rollout not properly initialized!")
        return
    
    # Test prompt generation step by step
    print(f"\n🔍 TESTING PROMPT GENERATION STEP BY STEP:")
    print("=" * 60)
    
    # Step 1: Test get_batch_llm_prompts
    print(f"\n   Step 1: Testing get_batch_llm_prompts...")
    try:
        batch_prompts = rollout.get_batch_llm_prompts(rollout.env_outs)
        print(f"   ✅ get_batch_llm_prompts succeeded")
        print(f"      DataProto type: {type(batch_prompts)}")
        print(f"      Has batch: {hasattr(batch_prompts, 'batch')}")
        print(f"      Has non_tensor_batch: {hasattr(batch_prompts, 'non_tensor_batch')}")
        print(f"      Has meta_info: {hasattr(batch_prompts, 'meta_info')}")
        
        if hasattr(batch_prompts, 'batch') and batch_prompts.batch is not None:
            print(f"      Batch keys: {list(batch_prompts.batch.keys())}")
            
            # Check each tensor in detail
            for key, tensor in batch_prompts.batch.items():
                if hasattr(tensor, 'shape'):
                    print(f"         {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                    
                    # Special checks for input_ids
                    if key == 'input_ids':
                        print(f"            Sample input_ids[0][:20]: {tensor[0][:20].tolist()}")
                        
                        # Check for padding-only sequences
                        for i in range(min(3, tensor.shape[0])):
                            seq = tensor[i]
                            non_pad_count = (seq != tokenizer.pad_token_id).sum().item()
                            total_count = seq.shape[0]
                            print(f"            Agent {i}: {non_pad_count}/{total_count} non-padding tokens")
                            
                            if non_pad_count == 0:
                                print(f"            ❌ AGENT {i} HAS ALL PADDING TOKENS!")
                            elif non_pad_count < 5:
                                print(f"            ⚠️  AGENT {i} HAS VERY FEW NON-PADDING TOKENS!")
                    
                    elif key == 'attention_mask':
                        # Check attention mask validity
                        for i in range(min(3, tensor.shape[0])):
                            seq = tensor[i]
                            valid_positions = seq.sum().item()
                            total_positions = seq.shape[0]
                            print(f"            Agent {i}: {valid_positions}/{total_positions} valid attention positions")
                            
                            if valid_positions == 0:
                                print(f"            ❌ AGENT {i} HAS NO VALID ATTENTION POSITIONS!")
                                
    except Exception as e:
        print(f"   ❌ get_batch_llm_prompts failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Test individual agent prompt generation
    print(f"\n   Step 2: Testing individual agent prompts...")
    for i in range(min(3, len(rollout.agents))):
        agent = rollout.agents[i]
        env_out = rollout.env_outs[i]
        
        print(f"\n      Agent {i} Individual Prompt Generation:")
        print(f"         Agent ID: {agent.agent_id}")
        print(f"         Group ID: {agent.group_id}")
        print(f"         Current Turn: {agent.cur_turn}")
        print(f"         Total Actions Consumed: {agent.total_actions_consumed}")
        print(f"         Max Turns: {agent.max_turns}")
        print(f"         Max Actions All Turns: {agent.max_actions_all_turns}")
        
        # Test agent's get_llm_prompts
        try:
            messages = agent.get_llm_prompts(env_out)
            print(f"         ✅ get_llm_prompts succeeded, {len(messages)} messages")
            
            # Print messages structure
            for j, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                content_preview = content[:100] + '...' if len(content) > 100 else content
                print(f"            Message {j}: {role} - '{content_preview}'")
                print(f"               Content length: {len(content)}")
            
            # Test chat template application
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Add answer format prompt
            enable_think = getattr(config_obj.rollout, 'enable_think', False)
            if enable_think:
                prompt_str += "<think>"
            else:
                prompt_str += "<answer>"
            
            print(f"         Final prompt length: {len(prompt_str)}")
            print(f"         Final prompt (first 200 chars): '{prompt_str[:200]}{'...' if len(prompt_str) > 200 else ''}'")
            
            # Test tokenization of this specific prompt
            debug_tokenization(tokenizer, prompt_str, f"Agent {i} Individual Prompt")
                
        except Exception as e:
            print(f"         ❌ Agent {i} prompt generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Test conversation state and history
    print(f"\n   Step 3: Testing conversation state and history...")
    for i in range(min(3, len(rollout.agents))):
        agent = rollout.agents[i]
        
        print(f"\n      Agent {i} Conversation State:")
        print(f"         Messages count: {len(agent.messages)}")
        print(f"         Trajectory history length: {len(agent.trajectory_history)}")
        
        # Print current messages
        for j, msg in enumerate(agent.messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_preview = content[:100] + '...' if len(content) > 100 else content
            print(f"            Message {j}: {role} - '{content_preview}'")
        
        # Check if agent is in valid state
        is_done = rollout.done_mask[i].item() if i < len(rollout.done_mask) else False
        print(f"         Agent done status: {is_done}")
        
        if agent.cur_turn >= agent.max_turns:
            print(f"         ⚠️  Agent has reached max turns ({agent.cur_turn}/{agent.max_turns})")
        
        if agent.total_actions_consumed >= agent.max_actions_all_turns:
            print(f"         ⚠️  Agent has consumed max actions ({agent.total_actions_consumed}/{agent.max_actions_all_turns})")
    
    print(f"\n   ✅ Prompt generation debugging completed")
    rollout.close()


def test_edge_case_scenarios():
    """Test edge cases that might cause empty prompts"""
    print("🔍 TESTING EDGE CASE SCENARIOS...")
    print("=" * 70)
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    # Test 1: Empty conversation (EXPECTED TO FAIL)
    print(f"\n   Test 1: Empty conversation handling...")
    print(f"      Note: This test EXPECTS to fail - testing tokenizer behavior with empty messages")
    try:
        messages = []
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        debug_tokenization(tokenizer, prompt_str, "Empty Conversation")
        print(f"   ⚠️  Unexpectedly succeeded - tokenizer handled empty messages")
    except Exception as e:
        print(f"   ✅ Expected failure: {e}")
        print(f"      This confirms tokenizer correctly rejects empty message lists")
    
    # Test 2: Very short conversation
    print(f"\n   Test 2: Very short conversation...")
    try:
        messages = [{"role": "user", "content": "Hi"}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        debug_tokenization(tokenizer, prompt_str, "Short Conversation")
    except Exception as e:
        print(f"   ❌ Short conversation test failed: {e}")
    
    # Test 3: Whitespace-only content
    print(f"\n   Test 3: Whitespace-only content...")
    try:
        messages = [{"role": "user", "content": "   \n\t  "}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        debug_tokenization(tokenizer, prompt_str, "Whitespace-only Content")
    except Exception as e:
        print(f"   ❌ Whitespace-only content test failed: {e}")
    
    # Test 4: Special tokens only
    print(f"\n   Test 4: Special tokens handling...")
    try:
        special_text = f"{tokenizer.pad_token}{tokenizer.eos_token}"
        debug_tokenization(tokenizer, special_text, "Special Tokens Only")
    except Exception as e:
        print(f"   ❌ Special tokens test failed: {e}")
    
    # Test 5: Long sequence truncation
    print(f"\n   Test 5: Long sequence truncation...")
    try:
        long_content = "This is a very long message. " * 200  # Make it long
        messages = [{"role": "user", "content": long_content}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        debug_tokenization(tokenizer, prompt_str, "Long Sequence")
    except Exception as e:
        print(f"   ❌ Long sequence test failed: {e}")
    
    print(f"\n   ✅ Edge case testing completed")
    print(f"      Note: Empty conversation failure is expected and confirms proper tokenizer behavior")


def test_full_rollout_with_debugging():
    """Test full rollout with step-by-step debugging"""
    print("🔍 TESTING FULL ROLLOUT WITH DEBUGGING...")
    print("=" * 70)
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    print(f"\n🚀 STARTING FULL ROLLOUT WITH STEP-BY-STEP DEBUGGING")
    print("=" * 60)
    
    # Reset rollout
    rollout._reset_batch_agents()
    
    # Run rollout turn by turn with debugging
    for turn in range(rollout.max_turns):
        print(f"\n   Turn {turn + 1}/{rollout.max_turns}")
        print("   " + "-" * 40)
        
        # Check if all agents are done
        done_count = rollout.done_mask.sum().item()
        print(f"      Done agents: {done_count}/{rollout.n_agents}")
        
        if rollout.done_mask.all():
            print(f"      All agents done, breaking early")
            break
        
        # Step 1: Generate batch prompts
        print(f"      Step 1: Generating batch prompts...")
        try:
            ensure_rollout_initialized(rollout)
            batch_prompts = rollout.get_batch_llm_prompts(rollout.env_outs)
            print(f"         ✅ Batch prompts generated successfully")
            
            # Validate prompts before sending to vLLM simulation
            print(f"         Validating prompts for vLLM compatibility...")
            
            if hasattr(batch_prompts, 'batch') and 'input_ids' in batch_prompts.batch:
                input_ids = batch_prompts.batch['input_ids']
                
                # Check each sequence for padding-only issue
                problematic_agents = []
                for i in range(input_ids.shape[0]):
                    seq = input_ids[i]
                    non_pad_count = (seq != tokenizer.pad_token_id).sum().item()
                    
                    if non_pad_count == 0:
                        problematic_agents.append(i)
                        print(f"         ❌ Agent {i}: ALL PADDING TOKENS (will cause IndexError)")
                    elif non_pad_count < 3:
                        print(f"         ⚠️  Agent {i}: Very few tokens ({non_pad_count})")
                
                if problematic_agents:
                    print(f"         ❌ Found {len(problematic_agents)} agents with problematic prompts!")
                    print(f"         This would cause IndexError in vLLM _pre_process_inputs")
                    
                    # Debug the problematic agents
                    for agent_idx in problematic_agents[:2]:  # Show first 2
                        print(f"\n         Debugging Agent {agent_idx}:")
                        agent = rollout.agents[agent_idx]
                        env_out = rollout.env_outs[agent_idx]
                        
                        print(f"            Agent state:")
                        print(f"               Current turn: {agent.cur_turn}")
                        print(f"               Done status: {rollout.done_mask[agent_idx].item()}")
                        print(f"               Messages count: {len(agent.messages)}")
                        print(f"               Trajectory length: {len(agent.trajectory_history)}")
                        
                        print(f"            Environment output:")
                        print(f"               Done: {env_out.done}")
                        print(f"               State length: {len(env_out.state)}")
                        print(f"               Reward: {env_out.reward}")
                        
                        # Try to regenerate prompt for this agent
                        try:
                            agent_messages = agent.get_llm_prompts(env_out)
                            print(f"               Generated {len(agent_messages)} messages")
                            
                            for j, msg in enumerate(agent_messages):
                                content = msg.get('content', '')
                                print(f"                  Message {j}: {msg.get('role')} - {len(content)} chars")
                                
                        except Exception as e:
                            print(f"               ❌ Failed to regenerate agent messages: {e}")
                
            else:
                print(f"         ❌ No input_ids found in batch prompts!")
                
        except Exception as e:
            print(f"         ❌ Batch prompt generation failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Step 2: Generate sequences (mocked)
        print(f"      Step 2: Generating LLM responses...")
        try:
            lm_outputs = rollout.generate_sequences(batch_prompts)
            print(f"         ✅ LLM responses generated successfully")
        except Exception as e:
            print(f"         ❌ LLM response generation failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Step 3: Process outputs
        print(f"      Step 3: Processing environment outputs...")
        try:
            rollout.env_outs = rollout.get_batch_env_outputs(lm_outputs)
            print(f"         ✅ Environment outputs processed successfully")
            
            # Check updated environment outputs
            new_done_count = rollout.done_mask.sum().item()
            print(f"         Updated done agents: {new_done_count}/{rollout.n_agents}")
            
        except Exception as e:
            print(f"         ❌ Environment output processing failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        rollout.step_cnt += 1
    
    print(f"\n   ✅ Full rollout debugging completed")
    print(f"      Total steps: {rollout.step_cnt}")
    print(f"      Final done agents: {rollout.done_mask.sum().item()}/{rollout.n_agents}")
    
    rollout.close()


def test_sync_multi_turn_rollout_creation():
    """Test SyncMultiTurnRollout creation"""
    print("🔍 Testing SyncMultiTurnRollout creation...")
    
    # Load real configuration
    config = load_config()
    config_obj = create_config_object(config)
    
    # Debug: Print detailed config structure
    print(f"   📊 DETAILED CONFIG DEBUG INFO:")
    print(f"      Config type: {type(config_obj)}")
    print(f"      Config top-level keys: {list(config_obj._original_dict.keys()) if hasattr(config_obj, '_original_dict') else 'NO_DICT'}")
    
    # Check required config values for sync_multi_turn_rollout
    required_configs = {
        'max_prompt_length': 'cfg.max_prompt_length',
        'rollout.truncation': 'cfg.rollout.truncation',
        'rollout.agent_group_num': 'cfg.rollout.agent_group_num',
        'rollout.agent_group_size': 'cfg.rollout.agent_group_size',
        'rollout.train': 'cfg.rollout.train',
        'rollout.enable_think': 'cfg.rollout.enable_think',
        'rollout.use_turn_scores': 'cfg.rollout.use_turn_scores',
        'rollout.enable_response_mask': 'cfg.rollout.enable_response_mask',
    }
    
    print(f"\n      📋 REQUIRED CONFIG CHECK:")
    for config_path, access_path in required_configs.items():
        try:
            if '.' in config_path:
                # Handle nested config access
                parts = config_path.split('.')
                current_obj = config_obj
                for part in parts:
                    current_obj = getattr(current_obj, part)
                value = current_obj
                status = "✅ FOUND"
            else:
                # Handle top-level config access
                value = getattr(config_obj, config_path)
                status = "✅ FOUND"
        except AttributeError as e:
            value = f"❌ MISSING: {str(e)}"
            status = "❌ MISSING"
        
        print(f"         {access_path}: {value} ({status})")
    
    # Check rollout config in detail
    print(f"\n      📋 ROLLOUT CONFIG DETAILS:")
    if hasattr(config_obj, 'rollout'):
        rollout_config = getattr(config_obj, 'rollout')
        print(f"         Type: {type(rollout_config)}")
        print(f"         Keys: {list(rollout_config._original_dict.keys()) if hasattr(rollout_config, '_original_dict') else 'NO_DICT'}")
        
        # Check each rollout attribute
        rollout_attrs = ['agent_group_num', 'agent_group_size', 'train', 'validation', 'truncation', 'enable_think', 'use_turn_scores', 'enable_response_mask']
        for attr in rollout_attrs:
            try:
                value = getattr(rollout_config, attr)
                print(f"         {attr}: {value} (✅ FOUND)")
            except AttributeError:
                print(f"         {attr}: ❌ MISSING")
    else:
        print(f"         ❌ NO ROLLOUT CONFIG FOUND!")
    
    # Check sokobanAgent config
    print(f"\n      📋 SOKOBAN AGENT CONFIG:")
    if hasattr(config_obj, 'sokobanAgent'):
        sokoban_config = getattr(config_obj, 'sokobanAgent')
        print(f"         Type: {type(sokoban_config)}")
        print(f"         Keys: {list(sokoban_config._original_dict.keys()) if hasattr(sokoban_config, '_original_dict') else 'NO_DICT'}")
        
        # Check agent_config nested structure
        if hasattr(sokoban_config, 'agent_config'):
            agent_config = sokoban_config.agent_config
            print(f"         agent_config type: {type(agent_config)}")
            print(f"         agent_config keys: {list(agent_config._original_dict.keys()) if hasattr(agent_config, '_original_dict') else 'NO_DICT'}")
            
            # Check max_turns specifically
            try:
                max_turns = agent_config.max_turns
                print(f"         max_turns: {max_turns} (✅ FOUND)")
            except AttributeError:
                print(f"         max_turns: ❌ MISSING")
        else:
            print(f"         ❌ NO AGENT_CONFIG FOUND!")
    else:
        print(f"         ❌ NO SOKOBAN AGENT CONFIG FOUND!")
    
    # Show what we have vs what we need
    print(f"\n      🔍 CONFIG AVAILABILITY SUMMARY:")
    print(f"         Available top-level keys: {list(config_obj._original_dict.keys()) if hasattr(config_obj, '_original_dict') else 'NONE'}")
    print(f"         Need for rollout: max_prompt_length, rollout.truncation, rollout.agent_group_num, rollout.agent_group_size, etc.")
    print(f"         Need for agents: sokobanAgent.agent_config.max_turns")
    
    # Create real tokenizer
    tokenizer = create_real_tokenizer()
    
    # Create mock actor worker group (only thing we mock)
    mock_actor_wg = create_mock_actor_wg()
    
    # Create rollout manager
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None  # Not used in our tests
    )
    
    # Verify creation - now use rollout config structure
    assert rollout.cfg == config_obj
    assert rollout.tokenizer == tokenizer
    assert rollout.actor_wg == mock_actor_wg
    
    # Calculate expected number of agents from agent_group_num * agent_group_size
    expected_n_agents = config['rollout']['agent_group_num'] * config['rollout']['agent_group_size']
    assert rollout.n_agents == expected_n_agents
    assert len(rollout.agents) == expected_n_agents
    assert rollout.step_cnt == 0
    
    print(f"✅ SyncMultiTurnRollout created successfully")
    print(f"   Number of agents: {rollout.n_agents} (calculated as {config['rollout']['agent_group_num']} groups × {config['rollout']['agent_group_size']} agents/group)")
    print(f"   Agent class: {rollout.agent_cls}")
    print(f"   Max turns: {rollout.max_turns}")
    print(f"   Step count: {rollout.step_cnt}")
    
    rollout.close()


def test_sync_multi_turn_rollout_full_rollout():
    """Test complete rollout process"""
    print("🔍 Testing complete rollout process...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    print(f"\n   🚀 STARTING FULL ROLLOUT")
    print("   " + "=" * 50)
    
    # Execute rollout
    final_env_outs = rollout.rollout()
    
    # Verify rollout results
    assert len(final_env_outs) == rollout.n_agents
    assert rollout.step_cnt > 0
    
    print(f"\n   📊 ROLLOUT RESULTS:")
    print(f"      Total steps: {rollout.step_cnt}")
    print(f"      Number of agents: {len(final_env_outs)}")
    
    for i, env_out in enumerate(final_env_outs):
        print(f"      Agent {i}: Done={env_out.done}, Reward={env_out.reward}")
        
        # Check agent trajectory
        agent = rollout.agents[i]
        print(f"         Trajectory length: {len(agent.trajectory_history)}")
        print(f"         Total actions consumed: {agent.total_actions_consumed}")
        print(f"         Current turn: {agent.cur_turn}")
    
    print(f"\n   ✅ Full rollout completed successfully")
    
    rollout.close()


def test_sync_multi_turn_rollout_final_states():
    """Test final rollout states collection"""
    print("🔍 Testing final rollout states collection...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    # Execute rollout
    final_env_outs = rollout.rollout()
    
    # Collect final rollout states
    final_rollout_states = rollout._collect_final_rollout_states()
    
    # Verify final states
    assert len(final_rollout_states) == rollout.n_agents
    
    print(f"\n   📊 FINAL ROLLOUT STATES:")
    print(f"      Number of agent states: {len(final_rollout_states)}")
    
    for i, state in enumerate(final_rollout_states):
        print(f"\n      Agent {i} Final State:")
        print(f"         Environment ID: {state['env_id']}")
        print(f"         Group ID: {state['group_id']}")
        print(f"         Tag: {state['tag']}")
        print(f"         History Length: {len(state['history'])}")
        print(f"         Penalty: {state.get('penalty', 0.0)}")
        
        # Print metrics
        print(f"         Metrics Values:")
        for key, value in state['metrics'].items():
            print(f"            {key}: {value}")
        
        # Print trajectory history
        print(f"         Trajectory History:")
        for j, step in enumerate(state['history']):
            print(f"            Step {j+1}:")
            print(f"               Actions Left: {step['actions_left']}")
            print(f"               Actions: {step['actions']}")
            print(f"               Reward: {step['reward']}")
            print(f"               Info: {step['info']}")
            print(f"               LLM Response: {step['llm_response']}")
            
            # Print state (first few lines)
            state_lines = step['state'].split('\n')[:3]
            print(f"               State (first 3 lines):")
            for line in state_lines:
                print(f"                  {line}")
    
    print(f"\n   ✅ Final rollout states collected successfully")
    
    rollout.close()


def test_sync_multi_turn_rollout_ppo_batch():
    """Test PPO batch building"""
    print("🔍 Testing PPO batch building...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    # Execute rollout
    final_env_outs = rollout.rollout()
    
    # Build PPO batch
    ppo_batch = rollout.build_ppo_batch()
    
    # Verify PPO batch
    assert ppo_batch is not None
    assert hasattr(ppo_batch, 'batch')
    assert hasattr(ppo_batch, 'non_tensor_batch')
    assert hasattr(ppo_batch, 'meta_info')
    
    # Check batch tensors
    batch_keys = ['input_ids', 'attention_mask', 'position_ids', 'responses', 'loss_mask', 'rm_scores', 'original_rm_scores']
    for key in batch_keys:
        assert key in ppo_batch.batch, f"Missing key: {key}"
    
    # Check non-tensor batch
    non_tensor_keys = ['env_ids', 'group_ids', 'messages_list']
    for key in non_tensor_keys:
        assert key in ppo_batch.non_tensor_batch, f"Missing non-tensor key: {key}"
    
    # Check meta info
    assert 'metrics' in ppo_batch.meta_info
    
    print(f"\n   📊 PPO BATCH STRUCTURE:")
    print(f"      Batch keys: {list(ppo_batch.batch.keys())}")
    print(f"      Non-tensor batch keys: {list(ppo_batch.non_tensor_batch.keys())}")
    print(f"      Meta info keys: {list(ppo_batch.meta_info.keys())}")
    
    # Print tensor shapes
    print(f"\n      Tensor Shapes:")
    for key, tensor in ppo_batch.batch.items():
        if hasattr(tensor, 'shape'):
            print(f"         {key}: {tensor.shape}")
        else:
            print(f"         {key}: {type(tensor)}")
    
    # Print metrics
    print(f"\n      Metrics:")
    for key, value in ppo_batch.meta_info['metrics'].items():
        print(f"         {key}: {value}")
    
    # Add detailed DataProto visualization
    print(f"\n   📊 DETAILED DATAPROTO VISUALIZATION:")
    print("   " + "=" * 50)
    
    # Check length
    try:
        print(f"      DataProto length: {len(ppo_batch)}")
    except:
        print(f"      DataProto length: Not available")
    
    # Access tensor data
    print(f"\n      Tensor Data (ppo_batch.batch):")
    print(f"         Type: {type(ppo_batch.batch)}")
    if hasattr(ppo_batch.batch, 'batch_size'):
        print(f"         Batch size: {ppo_batch.batch.batch_size}")
    
    for key, tensor in ppo_batch.batch.items():
        if hasattr(tensor, 'shape'):
            print(f"         {key}:")
            print(f"            Shape: {tensor.shape}")
            print(f"            Dtype: {tensor.dtype}")
            print(f"            Device: {tensor.device}")
            if tensor.numel() > 0:
                print(f"            Min/Max: {tensor.min().item():.4f} / {tensor.max().item():.4f}")
                print(f"            Sample values: {tensor.flatten()[:5].tolist()}")
        else:
            print(f"         {key}: {type(tensor)} - {tensor}")
    
    # Access non-tensor data
    print(f"\n      Non-tensor Data (ppo_batch.non_tensor_batch):")
    print(f"         Type: {type(ppo_batch.non_tensor_batch)}")
    for key, value in ppo_batch.non_tensor_batch.items():
        print(f"         {key}:")
        print(f"            Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"            Shape: {value.shape}")
        if hasattr(value, '__len__'):
            print(f"            Length: {len(value)}")
        if hasattr(value, 'dtype'):
            print(f"            Dtype: {value.dtype}")
        
        # Print sample values
        if key == 'env_ids':
            print(f"            Sample values: {value[:3] if len(value) > 3 else value}")
        elif key == 'group_ids':
            print(f"            Sample values: {value[:3] if len(value) > 3 else value}")
        elif key == 'messages_list':
            print(f"            Sample (first message):")
            if len(value) > 0 and len(value[0]) > 0:
                first_msg = value[0][0]
                print(f"               Role: {first_msg.get('role', 'N/A')}")
                content_preview = first_msg.get('content', '')[:100] + '...' if len(first_msg.get('content', '')) > 100 else first_msg.get('content', '')
                print(f"               Content: {content_preview}")
    
    # Access metadata
    print(f"\n      Metadata (ppo_batch.meta_info):")
    print(f"         Type: {type(ppo_batch.meta_info)}")
    for key, value in ppo_batch.meta_info.items():
        print(f"         {key}:")
        print(f"            Type: {type(value)}")
        if isinstance(value, dict):
            print(f"            Keys: {list(value.keys())}")
            for sub_key, sub_value in value.items():
                print(f"               {sub_key}: {sub_value} ({type(sub_value)})")
        else:
            print(f"            Value: {value}")
    
    print(f"\n   ✅ PPO batch built successfully")
    
    rollout.close()


def test_sync_multi_turn_rollout_complete():
    """Test complete rollout workflow with detailed output"""
    print("🔍 Testing complete rollout workflow...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    print(f"\n   🚀 COMPLETE ROLLOUT WORKFLOW")
    print("   " + "=" * 60)
    
    # Step 1: Execute rollout
    print(f"\n   Step 1: Executing rollout...")
    final_env_outs = rollout.rollout()
    print(f"   ✅ Rollout completed with {rollout.step_cnt} steps")
    
    # Step 2: Collect final rollout states
    print(f"\n   Step 2: Collecting final rollout states...")
    final_rollout_states = rollout._collect_final_rollout_states()
    print(f"   ✅ Collected states from {len(final_rollout_states)} agents")
    
    # Step 3: Build PPO batch
    print(f"\n   Step 3: Building PPO batch...")
    ppo_batch = rollout.build_ppo_batch()
    print(f"   ✅ PPO batch built successfully")
    
    # Print detailed final rollout states
    print(f"\n   📊 DETAILED FINAL ROLLOUT STATES:")
    print("   " + "=" * 60)
    
    for i, state in enumerate(final_rollout_states):
        print(f"\n   Agent {i} Complete State:")
        print(f"      Environment ID: {state['env_id']}")
        print(f"      Group ID: {state['group_id']}")
        print(f"      Tag: {state['tag']}")
        print(f"      History Length: {len(state['history'])}")
        print(f"      Penalty: {state.get('penalty', 0.0)}")
        
        print(f"\n      Metrics:")
        for key, value in state['metrics'].items():
            print(f"         {key}: {value}")
        
        print(f"\n      Complete Trajectory History:")
        for j, step in enumerate(state['history']):
            print(f"\n         Turn {j+1}:")
            print(f"            Actions Left: {step['actions_left']}")
            print(f"            Actions Executed: {step['actions']}")
            print(f"            Reward: {step['reward']}")
            print(f"            LLM Response: {step['llm_response']}")
            print(f"            LLM Raw Response: {step['llm_raw_response']}")
            
            # Print step info
            print(f"            Step Info:")
            for info_key, info_value in step['info'].items():
                print(f"               {info_key}: {info_value}")
            
            # Print environment state
            print(f"            Environment State:")
            state_lines = step['state'].split('\n')
            for line in state_lines:
                print(f"               {line}")
    
    # Print PPO batch summary
    print(f"\n   📊 PPO BATCH SUMMARY:")
    print("   " + "=" * 60)
    print(f"      Batch size: {len(final_rollout_states)}")
    print(f"      Total metrics: {len(ppo_batch.meta_info['metrics'])}")
    
    print(f"\n      Final Metrics Summary:")
    for key, value in ppo_batch.meta_info['metrics'].items():
        print(f"         {key}: {value}")
    
    print(f"\n   ✅ Complete rollout workflow test passed")
    
    rollout.close()


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting SyncMultiTurnRollout DEBUG Tests...")
        print()
        
        print("DEBUG Test 1: Prompt generation debugging")
        test_prompt_generation_debug()
        print()
        
        print("DEBUG Test 2: Edge case scenarios")
        test_edge_case_scenarios()
        print()
        
        print("DEBUG Test 3: Full rollout with debugging")
        test_full_rollout_with_debugging()
        print()
        
        # Run original tests too
        print("Original Test 1: Rollout creation")
        test_sync_multi_turn_rollout_creation()
        print()
        
        print("Original Test 2: Full rollout process")
        test_sync_multi_turn_rollout_full_rollout()
        print()
        
        print("Original Test 3: Final rollout states collection")
        test_sync_multi_turn_rollout_final_states()
        print()
        
        print("Original Test 4: PPO batch building")
        test_sync_multi_turn_rollout_ppo_batch()
        print()
        
        print("Original Test 5: Complete rollout workflow")
        test_sync_multi_turn_rollout_complete()
        print()
        
        print("=" * 70)
        print("🎉 All SyncMultiTurnRollout DEBUG tests completed!")
        print(f"✅ Test completed at {datetime.now()}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close the log file
        tee.close()
        sys.stdout = tee.stdout  # Restore original stdout
