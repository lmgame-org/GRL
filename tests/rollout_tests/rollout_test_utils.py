#!/usr/bin/env python3
"""
Rollout Test Utilities - Mock functions for testing rollout components
"""

import torch
from unittest.mock import Mock

def get_mock_lm_responses(batch_prompts):
    """
    Mock LLM responses function to replace actor_wg.generate_sequences()
    This function will be called directly in sync_multi_turn_rollout.py during testing
    
    Args:
        batch_prompts: DataProto containing batch of prompts
        
    Returns:
        DataProto: Mock LLM responses in the same format as real LLM output
    """
    # Predefined mock responses that cycle through
    mock_responses = [
        "<answer>Right || Down</answer>",
        "<answer>Left || Up</answer>", 
        "<answer>Down || Right</answer>",
        "<answer>Up || Left</answer>",
        "<answer>Right || Right</answer>",
        "<answer>Down || Down</answer>",
        "<answer>Left || Left</answer>",
        "<answer>Up || Up</answer>",
    ]
    
    # Get batch size from input
    if hasattr(batch_prompts, 'batch') and 'input_ids' in batch_prompts.batch:
        batch_size = batch_prompts.batch['input_ids'].shape[0]
    else:
        batch_size = 1
    
    print(f"   🤖 Mock LLM generating {batch_size} responses...")
    
    response_texts = []
    for i in range(batch_size):
        response = mock_responses[i % len(mock_responses)]
        print(f"      Agent {i}: {response}")
        response_texts.append(response)
    
    # Create a simple mock token tensor (we'll bypass tokenizer.batch_decode)
    response_tensor = torch.zeros((batch_size, 10), dtype=torch.long)
    
    # Import DataProto here to avoid circular imports
    try:
        from verl import DataProto
        
        # Create output in the same format as input
        output = DataProto()
        output.batch = {"responses": response_tensor}
        output.meta_info = getattr(batch_prompts, 'meta_info', {})
        output.non_tensor_batch = getattr(batch_prompts, 'non_tensor_batch', {})
        
    except ImportError:
        # Fallback mock DataProto
        class MockDataProto:
            def __init__(self):
                self.batch = {"responses": response_tensor}
                self.non_tensor_batch = {}
                self.meta_info = {}
                self.mock_response_texts = response_texts  # Store texts directly
        
        output = MockDataProto()
    
    return output


def create_mock_actor_wg():
    """
    Create a mock actor worker group that uses get_mock_lm_responses
    This is only used in tests, not in the actual rollout code
    """
    actor_wg = Mock()
    actor_wg.generate_sequences = get_mock_lm_responses
    return actor_wg


def create_mock_tokenizer_decode(original_tokenizer):
    """
    Create a mock tokenizer.batch_decode that returns our mock responses
    """
    # Store the original method
    original_batch_decode = original_tokenizer.batch_decode
    
    # Mock responses that cycle
    mock_responses = [
        "<answer>Right || Down</answer>",
        "<answer>Left || Up</answer>", 
        "<answer>Down || Right</answer>",
        "<answer>Up || Left</answer>",
        "<answer>Right || Right</answer>",
        "<answer>Down || Down</answer>",
        "<answer>Left || Left</answer>",
        "<answer>Up || Up</answer>",
    ]
    
    call_count = [0]  # Use list to make it mutable in closure
    
    def mock_batch_decode(token_ids, skip_special_tokens=True):
        batch_size = len(token_ids) if isinstance(token_ids, list) else token_ids.shape[0]
        responses = []
        
        for i in range(batch_size):
            response = mock_responses[(call_count[0] + i) % len(mock_responses)]
            responses.append(response)
        
        call_count[0] += batch_size
        return responses
    
    # Replace the method
    original_tokenizer.batch_decode = mock_batch_decode
    
    return original_tokenizer
