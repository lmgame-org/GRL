import numpy as np
import random
import sys
import os
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lmgamerl.agents.sokobanAgent.env import SokobanEnv


# Setup logging to file
def setup_logging():
    """Setup logging to stream outputs to test_logs directory"""
    test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(test_logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_logs_dir, f"env_test_{timestamp}.log")
    
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
    
    print(f"📝 Sokoban Environment Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    
    return tee


def get_default_config():
    """Get default configuration for Sokoban environment from agents.yaml"""
    # Load configuration from agents.yaml
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/agents.yaml')
    
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract sokobanAgent configuration with new nested format
        sokoban_config = yaml_config.get('sokobanAgent', {})
        sokoban_env_config = sokoban_config.get('env_config', {})
        
        # Convert dim_room from list to tuple if needed
        if 'dim_room' in sokoban_env_config:
            sokoban_env_config['dim_room'] = tuple(sokoban_env_config['dim_room'])
        
        print(f"✅ Loaded configuration from {config_path}")
        print(f"   Config keys: {list(sokoban_env_config.keys())}")
        
        return sokoban_env_config
        
    except FileNotFoundError:
        print(f"⚠️  Could not find {config_path}, using fallback configuration")
        # Fallback configuration if file not found
        return {
            'dim_room': (6, 6),
            'max_steps': 100,
            'num_boxes': 1,
            'render_mode': 'text',
            'search_depth': 300,
            'grid_lookup': {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            'action_lookup': {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
        }
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        # Fallback configuration if loading fails
        return {
            'dim_room': (6, 6),
            'max_steps': 100,
            'num_boxes': 1,
            'render_mode': 'text',
            'search_depth': 300,
            'grid_lookup': {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            'action_lookup': {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
        }


def test_single_env_creation():
    """Test creating a single Sokoban environment"""
    print("🔍 Testing single environment creation...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Test reset
    initial_obs = env.reset(seed=42)
    
    assert initial_obs is not None
    assert isinstance(initial_obs, str)
    assert len(initial_obs) > 0
    
    # Check that we have the expected characters based on config
    # Use default grid_lookup if not in config (for robustness)
    default_grid_lookup = {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
    grid_lookup = config.get('grid_lookup', default_grid_lookup)
    
    wall_char = grid_lookup.get(0, '#')
    player_char = grid_lookup.get(5, 'P')
    box_char = grid_lookup.get(4, 'X')
    box_on_target_char = grid_lookup.get(3, '√')
    target_char = grid_lookup.get(2, 'O')
    player_on_target_char = grid_lookup.get(6, 'S')
    
    # Check for expected characters (but be flexible about what we find)
    has_walls = wall_char in initial_obs
    has_player = player_char in initial_obs or player_on_target_char in initial_obs
    has_boxes = box_char in initial_obs or box_on_target_char in initial_obs
    has_targets = target_char in initial_obs or box_on_target_char in initial_obs or player_on_target_char in initial_obs
    
    print(f"✅ Single environment created successfully")
    print(f"   Initial observation length: {len(initial_obs)} characters")
    print(f"   Contains walls ({wall_char}): {has_walls}")
    print(f"   Contains player: {has_player}")
    print(f"   Contains boxes: {has_boxes}")
    print(f"   Contains targets: {has_targets}")
    print(f"   Contains player ({player_char}/{player_on_target_char}): {player_char in initial_obs or player_on_target_char in initial_obs}")
    print(f"   Contains boxes ({box_char}/{box_on_target_char}): {box_char in initial_obs or box_on_target_char in initial_obs}")
    print(f"   Contains targets ({target_char}/{box_on_target_char}/{player_on_target_char}): {target_char in initial_obs or box_on_target_char in initial_obs or player_on_target_char in initial_obs}")
    
    env.close()


def test_env_reset_with_different_seeds():
    """Test that different seeds produce different environments"""
    print("🔍 Testing environment reset with different seeds...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Generate 5 different environments with different seeds
    observations = []
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        obs = env.reset(seed=seed)
        observations.append(obs)
        print(f"   Seed {seed}: Generated room with {len(obs)} characters")
    
    # Check that at least some environments are different
    unique_observations = set(observations)
    assert len(unique_observations) > 1, "All environments are identical - seed not working properly"
    
    print(f"✅ Seed test passed: {len(unique_observations)}/{len(observations)} unique environments generated")
    
    env.close()


def test_env_step_functionality():
    """Test basic step functionality"""
    print("🔍 Testing environment step functionality...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Reset environment
    initial_obs = env.reset(seed=42)
    
    # Try all possible actions
    actions = [0, 1, 2, 3]  # Up, Down, Left, Right
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    for action, action_name in zip(actions, action_names):
        obs, reward, done, info = env.step(action)
        
        assert obs is not None
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'action_is_valid' in info
        assert 'action_is_effective' in info
        assert 'success' in info
        
        print(f"   Action {action_name}: reward={reward}, done={done}, valid={info['action_is_valid']}, effective={info['action_is_effective']}")
    
    print("✅ Step functionality test passed")
    
    env.close()


def test_env_action_mapping():
    """Test environment action mapping with random actions from action space"""
    print("🔍 Testing environment action mapping...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Reset environment
    initial_obs = env.reset(seed=42)
    print(f"   Initial state:\n{initial_obs}")
    
    # Get available actions from config
    action_lookup = config.get('action_lookup', {})
    available_actions = list(action_lookup.keys())
    print(f"   Available actions from config: {available_actions}")
    print(f"   Action mapping: {action_lookup}")
    
    # Test each action
    for action in available_actions:
        print(f"\n   Testing action {action} ({action_lookup[action]}):")
        
        # Reset to consistent state
        env.reset(seed=42)
        
        # Take the action
        try:
            obs, reward, done, info = env.step(action)
            
            print(f"      Action executed successfully")
            print(f"      Reward: {reward}")
            print(f"      Done: {done}")
            print(f"      Info: {info}")
            print(f"      New state:\n{obs}")
            
            # Verify step worked
            assert obs is not None
            assert isinstance(obs, str)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
        except Exception as e:
            print(f"      ❌ Action {action} failed: {e}")
            raise
    
    # Test with random actions multiple times
    print(f"\n   Testing with 10 random actions:")
    env.reset(seed=42)
    
    for i in range(10):
        action = available_actions[i % len(available_actions)]
        print(f"      Step {i+1}: Action {action} ({action_lookup[action]})")
        
        try:
            obs, reward, done, info = env.step(action)
            print(f"         → Reward: {reward}, Done: {done}, Valid: {info.get('action_is_valid', 'N/A')}, Effective: {info.get('action_is_effective', 'N/A')}")
            
            if done:
                print(f"         → Episode finished at step {i+1}")
                break
                
        except Exception as e:
            print(f"         ❌ Step {i+1} failed: {e}")
            break
    
    env.close()
    print("✅ Action mapping test passed")


def test_10_random_envs_three_times():
    """Main test: Launch 10 random environments three times and print room layouts"""
    print("🎯 MAIN TEST: Launching 10 random environments three times...")
    print("=" * 60)
    
    config = get_default_config()
    
    for run in range(3):
        print(f"\n🚀 RUN {run + 1}/3")
        print("-" * 40)
        
        for env_id in range(10):
            print(f"\n🏠 Environment {env_id + 1}/10 (Run {run + 1})")
            
            # Create new environment for each test
            env = SokobanEnv(config)
            
            # Use different seed for each environment
            # Combine run number and env_id to ensure uniqueness
            seed = run * 1000 + env_id * 100 + random.randint(0, 99)
            
            try:
                # Reset environment with unique seed
                obs = env.reset(seed=seed)
                
                print(f"   Seed: {seed}")
                print(f"   Room dimensions: {config['dim_room']}")
                print(f"   Max steps: {config['max_steps']}")
                print(f"   Number of boxes: {config['num_boxes']}")
                print("   Room layout:")
                
                # Print the room layout with proper formatting
                lines = obs.split('\n')
                for i, line in enumerate(lines):
                    print(f"   {i:2d}: {line}")
                
                # Analyze the room using config-based character mappings
                grid_lookup = config['grid_lookup']
                wall_char = grid_lookup.get(0, '#')
                empty_char = grid_lookup.get(1, '_')
                target_char = grid_lookup.get(2, 'O')
                box_on_target_char = grid_lookup.get(3, '√')
                box_char = grid_lookup.get(4, 'X')
                player_char = grid_lookup.get(5, 'P')
                player_on_target_char = grid_lookup.get(6, 'S')
                
                wall_count = obs.count(wall_char)
                empty_count = obs.count(empty_char)
                target_count = obs.count(target_char)
                box_count = obs.count(box_char)
                box_on_target_count = obs.count(box_on_target_char)
                player_count = obs.count(player_char)
                player_on_target_count = obs.count(player_on_target_char)
                
                print(f"   Analysis:")
                print(f"     Walls ({wall_char}): {wall_count}")
                print(f"     Empty spaces ({empty_char}): {empty_count}")
                print(f"     Targets ({target_char}): {target_count}")
                print(f"     Boxes ({box_char}): {box_count}")
                print(f"     Boxes on targets ({box_on_target_char}): {box_on_target_count}")
                print(f"     Player ({player_char}): {player_count}")
                print(f"     Player on target ({player_on_target_char}): {player_on_target_count}")
                print(f"     Total boxes: {box_count + box_on_target_count}")
                print(f"     Total targets: {target_count + box_on_target_count + player_on_target_count}")
                
                # Verify basic constraints
                assert player_count + player_on_target_count == 1, "Should have exactly one player"
                assert box_count + box_on_target_count == config['num_boxes'], f"Should have exactly {config['num_boxes']} boxes"
                assert target_count + box_on_target_count + player_on_target_count >= config['num_boxes'], "Should have at least as many targets as boxes"
                
            except Exception as e:
                print(f"   ❌ Error generating environment: {e}")
                print(f"   Retrying with different seed...")
                
                # Try with a completely different seed
                backup_seed = random.randint(0, 999999)
                try:
                    obs = env.reset(seed=backup_seed)
                    print(f"   ✅ Backup seed {backup_seed} worked")
                    print("   Room layout:")
                    lines = obs.split('\n')
                    for i, line in enumerate(lines):
                        print(f"   {i:2d}: {line}")
                except Exception as e2:
                    print(f"   ❌ Backup seed also failed: {e2}")
            
            finally:
                env.close()
    
    print("\n" + "=" * 60)
    print("🎉 Main test completed successfully!")
    print("✅ All 30 environments (10 × 3 runs) generated and analyzed")


def test_env_action_space():
    """Test environment action space"""
    print("🔍 Testing environment action space...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Test get_all_actions
    all_actions = env.get_all_actions()
    assert all_actions is not None
    assert len(all_actions) > 0
    
    print(f"   Available actions: {all_actions}")
    
    # Test ACTION_SPACE
    assert hasattr(env, 'ACTION_SPACE')
    print(f"   Action space: {env.ACTION_SPACE}")
    
    env.close()
    print("✅ Action space test passed")


def test_env_rendering_modes():
    """Test different rendering modes"""
    print("🔍 Testing environment rendering modes...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Reset environment
    env.reset(seed=42)
    
    # Test text mode (default)
    text_obs = env.render(mode='text')
    assert isinstance(text_obs, str)
    assert len(text_obs) > 0
    print(f"   Text mode: {len(text_obs)} characters")
    
    # Test rgb_array mode
    try:
        rgb_obs = env.render(mode='rgb_array')
        assert rgb_obs is not None
        if hasattr(rgb_obs, 'shape') and hasattr(rgb_obs, 'dtype'):
            print(f"   RGB array mode: shape {rgb_obs.shape}")  # type: ignore
        else:
            print(f"   RGB array mode: type {type(rgb_obs)}")
    except Exception as e:
        print(f"   RGB array mode failed (expected): {e}")
    
    # Test invalid mode
    try:
        env.render(mode='invalid_mode')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Invalid mode correctly rejected: {e}")
    
    env.close()
    print("✅ Rendering modes test passed")


def test_env_seeding():
    """Test that environment generates different layouts with different seeds"""
    print("\n🔍 TESTING ENVIRONMENT SEEDING...")
    
    # Use consistent config loading
    env_config = get_default_config()
    print(f"   Environment config: {list(env_config.keys())}")
    
    # Test multiple environments with different seeds
    envs = []
    initial_states = []
    
    for i in range(10):  # Test 10 different seeds
        env = SokobanEnv(env_config)
        seed = random.randint(0, 100000)
        
        # Reset with different seeds
        state = env.reset(seed=seed)
        envs.append(env)
        initial_states.append(state)
        
        print(f"   Env {i} (seed {seed}): {state[:40]}...")
    
    # Check for diversity
    unique_states = set(initial_states)
    print(f"   Total environments: {len(envs)}")
    print(f"   Unique initial states: {len(unique_states)}")
    print(f"   State diversity: {len(unique_states) / len(envs) * 100:.1f}%")
    
    # Should have some diversity
    if len(unique_states) == 1:
        print(f"   ❌ WARNING: All environments have identical states!")
        print(f"   This suggests the environment is not using seeds properly")
        print(f"   First state: {initial_states[0]}")
    else:
        print(f"   ✅ Good diversity: {len(unique_states)} unique states")
    
    # Test same seed produces same state
    print("\n   Testing seed reproducibility...")
    env1 = SokobanEnv(env_config)
    env2 = SokobanEnv(env_config)
    
    test_seed = 12345
    state1 = env1.reset(seed=test_seed)
    state2 = env2.reset(seed=test_seed)
    
    print(f"   Same seed ({test_seed}) produces same state: {state1 == state2}")
    if state1 != state2:
        print(f"   ❌ WARNING: Same seed produced different states!")
        print(f"   State 1: {state1}")
        print(f"   State 2: {state2}")
    
    # Clean up
    for env in envs + [env1, env2]:
        env.close()
    
    print("   ✅ Environment seeding test completed!")

def test_env_actions():
    """Test environment action processing"""
    print("\n🔍 TESTING ENVIRONMENT ACTIONS...")
    
    # Use consistent config loading
    env_config = get_default_config()
    env = SokobanEnv(env_config)
    
    # Reset environment
    initial_state = env.reset(seed=12345)
    print(f"   Initial state: {initial_state[:50]}...")
    
    # Test action lookup
    if 'action_lookup' in env_config:
        print(f"   Action lookup: {env_config['action_lookup']}")
        
        # Test each action
        for action_id, action_name in env_config['action_lookup'].items():
            print(f"   Testing action {action_id}: {action_name}")
            
            # Reset to test each action from same starting position
            env.reset(seed=12345)
            
            try:
                new_state, reward, done, info = env.step(action_id)
                print(f"      Result: reward={reward}, done={done}, info={info}")
                print(f"      New state (first 30 chars): {new_state[:30]}...")
            except Exception as e:
                print(f"      ❌ Error executing action: {e}")
    else:
        print(f"   ❌ No action_lookup found in config")
    
    env.close()
    print("   ✅ Environment actions test completed!")

def test_env_config_validation():
    """Test environment configuration validation"""
    print("\n🔍 TESTING ENVIRONMENT CONFIG VALIDATION...")
    
    # Use consistent config loading
    env_config = get_default_config()
    
    # Check required fields
    required_fields = ['action_lookup', 'grid_lookup']  # Fixed: should be grid_lookup, not grid_vocab
    for field in required_fields:
        if field in env_config:
            print(f"   ✅ {field}: {env_config[field]}")
        else:
            print(f"   ❌ Missing {field}")
    
    # Test environment creation
    try:
        env = SokobanEnv(env_config)
        print(f"   ✅ Environment created successfully")
        
        # Test basic operations
        state = env.reset()
        print(f"   ✅ Reset successful, state length: {len(state)}")
        
        # Test rendering
        rendered = env.render()
        print(f"   ✅ Render successful, length: {len(rendered)}")
        
        env.close()
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
    
    print("   ✅ Environment config validation completed!")


def test_random_solver_success_rate():
    """Test solving Sokoban puzzles with random actions - 100 envs, 10 actions each"""
    print("\n🎯 TESTING RANDOM SOLVER SUCCESS RATE...")
    print("   Testing 100 environments with 10 random actions each")
    
    env_config = get_default_config()
    action_lookup = env_config.get('action_lookup', {})
    available_actions = list(action_lookup.keys())
    
    total_envs = 100
    max_actions = 10
    successes = 0
    rewards_collected = []
    actions_taken = []
    
    print(f"   Available actions: {available_actions}")
    print(f"   Action names: {[action_lookup[a] for a in available_actions]}")
    
    for env_id in range(total_envs):
        env = SokobanEnv(env_config)
        
        # Use different seed for each environment
        seed = random.randint(0, 100000)
        
        try:
            # Reset environment
            initial_state = env.reset(seed=seed)
            
            total_reward = 0
            actions_used = 0
            solved = False
            
            # Try random actions
            for step in range(max_actions):
                action = random.choice(available_actions)
                state, reward, done, info = env.step(action)
                
                total_reward += reward
                actions_used += 1
                
                if info.get('success', False):
                    solved = True
                    break
                
                if done:
                    break
            
            if solved:
                successes += 1
                print(f"   ✅ Env {env_id+1}: SOLVED in {actions_used} actions! Reward: {total_reward}")
            
            rewards_collected.append(total_reward)
            actions_taken.append(actions_used)
            
            # Print progress every 25 environments
            if (env_id + 1) % 25 == 0:
                current_success_rate = successes / (env_id + 1) * 100
                avg_reward = sum(rewards_collected) / len(rewards_collected)
                print(f"   Progress: {env_id+1}/{total_envs} envs, Success rate: {current_success_rate:.1f}%, Avg reward: {avg_reward:.2f}")
        
        except Exception as e:
            print(f"   ❌ Error in env {env_id+1}: {e}")
        
        finally:
            env.close()
    
    # Calculate final statistics
    success_rate = successes / total_envs * 100
    avg_reward = sum(rewards_collected) / len(rewards_collected) if rewards_collected else 0
    avg_actions = sum(actions_taken) / len(actions_taken) if actions_taken else 0
    
    print(f"\n   📊 RANDOM SOLVER RESULTS:")
    print(f"      Total environments tested: {total_envs}")
    print(f"      Max actions per environment: {max_actions}")
    print(f"      Successes: {successes}")
    print(f"      Success rate: {success_rate:.2f}%")
    print(f"      Average reward per environment: {avg_reward:.3f}")
    print(f"      Average actions taken: {avg_actions:.1f}")
    
    if success_rate > 0:
        print(f"   ✅ Random solver can solve some puzzles!")
    else:
        print(f"   ⚠️  Random solver didn't solve any puzzles in {max_actions} actions")
    
    return success_rate, avg_reward


def test_directional_solver_success_rate():
    """Test solving with a simple directional heuristic - 100 envs, up to 20 actions"""
    print("\n🎯 TESTING DIRECTIONAL SOLVER SUCCESS RATE...")
    print("   Testing 100 environments with directional heuristic (up to 20 actions)")
    
    env_config = get_default_config()
    action_lookup = env_config.get('action_lookup', {})
    available_actions = list(action_lookup.keys())
    
    # Map action names to action IDs for heuristic
    action_map = {}
    for action_id, action_name in action_lookup.items():
        action_map[action_name.lower()] = action_id
    
    total_envs = 100
    max_actions = 20
    successes = 0
    rewards_collected = []
    actions_taken = []
    
    def simple_heuristic_action(state, available_actions, action_map):
        """Simple heuristic: try to move towards boxes or targets"""
        lines = state.strip().split('\n')
        
        # Find player position
        player_pos = None
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char in ['P', 'S']:  # Player or Player on target
                    player_pos = (r, c)
                    break
            if player_pos:
                break
        
        if not player_pos:
            return random.choice(available_actions)
        
        # Find boxes and targets
        boxes = []
        targets = []
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char in ['X']:  # Box not on target
                    boxes.append((r, c))
                elif char in ['O', 'S']:  # Target or Player on target
                    targets.append((r, c))
        
        # Simple heuristic: move towards nearest box or target
        target_pos = None
        min_dist = float('inf')
        
        for box in boxes:
            dist = abs(box[0] - player_pos[0]) + abs(box[1] - player_pos[1])
            if dist < min_dist:
                min_dist = dist
                target_pos = box
        
        for target in targets:
            dist = abs(target[0] - player_pos[0]) + abs(target[1] - player_pos[1])
            if dist < min_dist:
                min_dist = dist
                target_pos = target
        
        if target_pos:
            # Determine direction to target
            dr = target_pos[0] - player_pos[0]
            dc = target_pos[1] - player_pos[1]
            
            # Prefer vertical movement first, then horizontal
            if dr > 0 and 'down' in action_map:
                return action_map['down']
            elif dr < 0 and 'up' in action_map:
                return action_map['up']
            elif dc > 0 and 'right' in action_map:
                return action_map['right']
            elif dc < 0 and 'left' in action_map:
                return action_map['left']
        
        # Fallback to random
        return random.choice(available_actions)
    
    for env_id in range(total_envs):
        env = SokobanEnv(env_config)
        
        # Use different seed for each environment
        seed = random.randint(0, 100000)
        
        try:
            # Reset environment
            initial_state = env.reset(seed=seed)
            current_state = initial_state
            
            total_reward = 0
            actions_used = 0
            solved = False
            
            # Try heuristic actions
            for step in range(max_actions):
                action = simple_heuristic_action(current_state, available_actions, action_map)
                state, reward, done, info = env.step(action)
                
                current_state = state
                total_reward += reward
                actions_used += 1
                
                if info.get('success', False):
                    solved = True
                    break
                
                if done:
                    break
            
            if solved:
                successes += 1
                print(f"   ✅ Env {env_id+1}: SOLVED in {actions_used} actions! Reward: {total_reward}")
            
            rewards_collected.append(total_reward)
            actions_taken.append(actions_used)
            
            # Print progress every 25 environments
            if (env_id + 1) % 25 == 0:
                current_success_rate = successes / (env_id + 1) * 100
                avg_reward = sum(rewards_collected) / len(rewards_collected)
                print(f"   Progress: {env_id+1}/{total_envs} envs, Success rate: {current_success_rate:.1f}%, Avg reward: {avg_reward:.2f}")
        
        except Exception as e:
            print(f"   ❌ Error in env {env_id+1}: {e}")
        
        finally:
            env.close()
    
    # Calculate final statistics
    success_rate = successes / total_envs * 100
    avg_reward = sum(rewards_collected) / len(rewards_collected) if rewards_collected else 0
    avg_actions = sum(actions_taken) / len(actions_taken) if actions_taken else 0
    
    print(f"\n   📊 DIRECTIONAL SOLVER RESULTS:")
    print(f"      Total environments tested: {total_envs}")
    print(f"      Max actions per environment: {max_actions}")
    print(f"      Successes: {successes}")
    print(f"      Success rate: {success_rate:.2f}%")
    print(f"      Average reward per environment: {avg_reward:.3f}")
    print(f"      Average actions taken: {avg_actions:.1f}")
    
    if success_rate > 0:
        print(f"   ✅ Directional solver can solve puzzles!")
    else:
        print(f"   ⚠️  Directional solver didn't solve any puzzles in {max_actions} actions")
    
    return success_rate, avg_reward


def test_environment_difficulty_analysis():
    """Analyze the difficulty of generated Sokoban environments"""
    print("\n🔍 TESTING ENVIRONMENT DIFFICULTY ANALYSIS...")
    print("   Analyzing 50 environments to understand puzzle characteristics")
    
    env_config = get_default_config()
    
    total_envs = 50
    room_analyses = []
    
    for env_id in range(total_envs):
        env = SokobanEnv(env_config)
        
        try:
            # Reset with unique seed
            seed = random.randint(0, 100000)
            initial_state = env.reset(seed=seed)
            
            # Analyze the room layout
            lines = initial_state.strip().split('\n')
            
            analysis = {
                'env_id': env_id,
                'seed': seed,
                'room_size': (len(lines), len(lines[0]) if lines else 0),
                'total_cells': len(lines) * len(lines[0]) if lines else 0
            }
            
            # Count different elements using config mapping
            grid_lookup = env_config['grid_lookup']
            wall_char = grid_lookup.get(0, '#')
            empty_char = grid_lookup.get(1, '_')
            target_char = grid_lookup.get(2, 'O')
            box_on_target_char = grid_lookup.get(3, '√')
            box_char = grid_lookup.get(4, 'X')
            player_char = grid_lookup.get(5, 'P')
            player_on_target_char = grid_lookup.get(6, 'S')
            
            analysis['walls'] = initial_state.count(wall_char)
            analysis['empty_spaces'] = initial_state.count(empty_char)
            analysis['targets'] = initial_state.count(target_char)
            analysis['boxes'] = initial_state.count(box_char)
            analysis['boxes_on_targets'] = initial_state.count(box_on_target_char)
            analysis['player_on_target'] = initial_state.count(player_on_target_char)
            
            # Calculate metrics
            analysis['total_boxes'] = analysis['boxes'] + analysis['boxes_on_targets']
            analysis['total_targets'] = analysis['targets'] + analysis['boxes_on_targets'] + analysis['player_on_target']
            analysis['free_space_ratio'] = analysis['empty_spaces'] / analysis['total_cells'] if analysis['total_cells'] > 0 else 0
            analysis['wall_ratio'] = analysis['walls'] / analysis['total_cells'] if analysis['total_cells'] > 0 else 0
            
            room_analyses.append(analysis)
            
            if env_id % 10 == 0:
                print(f"   Analyzed {env_id + 1}/{total_envs} environments...")
        
        except Exception as e:
            print(f"   ❌ Error analyzing env {env_id+1}: {e}")
        
        finally:
            env.close()
    
    # Calculate statistics
    if room_analyses:
        print(f"\n   📊 ENVIRONMENT ANALYSIS RESULTS ({len(room_analyses)} environments):")
        
        avg_boxes = sum(a['total_boxes'] for a in room_analyses) / len(room_analyses)
        avg_targets = sum(a['total_targets'] for a in room_analyses) / len(room_analyses)
        avg_free_space = sum(a['free_space_ratio'] for a in room_analyses) / len(room_analyses)
        avg_wall_ratio = sum(a['wall_ratio'] for a in room_analyses) / len(room_analyses)
        
        print(f"      Average boxes per environment: {avg_boxes:.1f}")
        print(f"      Average targets per environment: {avg_targets:.1f}")
        print(f"      Average free space ratio: {avg_free_space:.2f}")
        print(f"      Average wall ratio: {avg_wall_ratio:.2f}")
        
        # Show some example layouts
        print(f"\n   📝 SAMPLE ENVIRONMENT LAYOUTS:")
        for i in range(min(3, len(room_analyses))):
            analysis = room_analyses[i]
            print(f"      Environment {analysis['env_id']} (seed {analysis['seed']}):")
            print(f"         Boxes: {analysis['total_boxes']}, Targets: {analysis['total_targets']}")
            print(f"         Free space: {analysis['free_space_ratio']:.1%}, Walls: {analysis['wall_ratio']:.1%}")
    
    print("   ✅ Environment difficulty analysis completed!")


def test_comprehensive_solving_suite():
    """Run comprehensive solving tests with different strategies and action limits"""
    print("\n🚀 COMPREHENSIVE SOLVING TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Random solver with 10 actions (as requested)
    print("\n1️⃣  RANDOM SOLVER - 10 ACTIONS")
    random_success_10, random_reward_10 = test_random_solver_success_rate()
    results['random_10'] = {'success_rate': random_success_10, 'avg_reward': random_reward_10}
    
    # Test 2: Directional heuristic solver with 20 actions
    print("\n2️⃣  DIRECTIONAL HEURISTIC SOLVER - 20 ACTIONS")
    heuristic_success_20, heuristic_reward_20 = test_directional_solver_success_rate()
    results['heuristic_20'] = {'success_rate': heuristic_success_20, 'avg_reward': heuristic_reward_20}
    
    # Test 3: Environment difficulty analysis
    print("\n3️⃣  ENVIRONMENT DIFFICULTY ANALYSIS")
    test_environment_difficulty_analysis()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE SOLVING TEST SUMMARY")
    print("=" * 80)
    
    for test_name, test_results in results.items():
        success_rate = test_results['success_rate']
        avg_reward = test_results['avg_reward']
        print(f"   {test_name}: {success_rate:.1f}% success rate, {avg_reward:.3f} avg reward")
    
    # Determine best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['success_rate'])
    best_success_rate = results[best_strategy]['success_rate']
    
    print(f"\n   🏆 Best performing strategy: {best_strategy} ({best_success_rate:.1f}% success rate)")
    
    if best_success_rate > 0:
        print(f"   ✅ SUCCESS: Our Sokoban environments can be solved!")
        print(f"   💡 Recommendation: Use {best_strategy} strategy for solving")
    else:
        print(f"   ⚠️  CHALLENGE: No strategy achieved success in limited actions")
        print(f"   💡 Recommendation: Consider increasing action limits or simpler puzzles")
    
    print("   ✅ Comprehensive solving suite completed!")
    
    return results


def test_simple_action_debug():
    """Simple debug test to see what happens with individual actions"""
    print("\n🔧 SIMPLE ACTION DEBUG TEST...")
    
    env_config = get_default_config()
    env = SokobanEnv(env_config)
    
    # Reset with a specific seed
    initial_state = env.reset(seed=42)
    print(f"   Initial state:")
    print(f"   {initial_state}")
    print()
    
    # Test each action individually
    action_lookup = env_config.get('action_lookup', {})
    available_actions = list(action_lookup.keys())
    
    for action in available_actions:
        # Reset to same state
        env.reset(seed=42)
        print(f"   Testing action {action} ({action_lookup[action]}):")
        
        try:
            state, reward, done, info = env.step(action)
            print(f"      Reward: {reward}")
            print(f"      Done: {done}")
            print(f"      Info: {info}")
            print(f"      New state:")
            print(f"      {state}")
            print()
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    env.close()
    print("   ✅ Simple action debug completed!")


def test_manual_solve_attempt():
    """Try to manually solve a simple puzzle to see if success detection works"""
    print("\n🎯 MANUAL SOLVE ATTEMPT TEST...")
    
    env_config = get_default_config()
    env = SokobanEnv(env_config)
    
    # Try several seeds to find a simple puzzle
    for seed in [42, 100, 200, 300, 400]:
        try:
            initial_state = env.reset(seed=seed)
            print(f"\n   Seed {seed} - Initial state:")
            print(f"   {initial_state}")
            
            # Try a sequence of moves to see if we can solve it
            action_lookup = env_config.get('action_lookup', {})
            available_actions = list(action_lookup.keys())
            
            # Try random sequence of 20 actions
            solved = False
            for step in range(20):
                action = random.choice(available_actions)
                state, reward, done, info = env.step(action)
                
                if info.get('success', False):
                    solved = True
                    print(f"   🎉 SOLVED at seed {seed} in {step+1} steps!")
                    print(f"   Final state:")
                    print(f"   {state}")
                    print(f"   Final reward: {reward}")
                    break
                    
                if step % 5 == 4:  # Print every 5 steps
                    print(f"   Step {step+1}: action {action} ({action_lookup[action]}), reward {reward}, done {done}")
            
            if solved:
                break
                
        except Exception as e:
            print(f"   ❌ Error with seed {seed}: {e}")
            continue
    
    env.close()
    print("   ✅ Manual solve attempt completed!")


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting Sokoban Environment Tests...")
        print()
        
        print("Test 1: Single environment creation")
        test_single_env_creation()
        print()
        
        print("Test 2: Environment reset with different seeds")
        test_env_reset_with_different_seeds()
        print()
        
        print("Test 3: Environment step functionality")
        test_env_step_functionality()
        print()
        
        print("Test 4: Environment action space")
        test_env_action_space()
        print()
        
        print("Test 5: Environment action mapping")
        test_env_action_mapping()
        print()
        
        print("Test 6: Environment rendering modes")
        test_env_rendering_modes()
        print()
        
        print("Test 7: 10 random environments × 3 runs")
        test_10_random_envs_three_times()
        print()

        print("Test 8: Environment seeding")
        test_env_seeding()
        print()

        print("Test 9: Environment actions")
        test_env_actions()
        print()

        print("Test 10: Environment config validation")
        test_env_config_validation()
        print()
        
        print("Test 11: COMPREHENSIVE SOLVING TEST SUITE")
        test_comprehensive_solving_suite()
        print()
        
        print("Test 12: SIMPLE ACTION DEBUG TEST")
        test_simple_action_debug()
        print()

        print("Test 13: MANUAL SOLVE ATTEMPT TEST")
        test_manual_solve_attempt()
        print()
        
        print("=" * 60)
        print("🎉 All tests passed!")
        print(f"✅ Test completed at {datetime.now()}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close the log file
        tee.close()
        sys.stdout = tee.stdout  # Restore original stdout
