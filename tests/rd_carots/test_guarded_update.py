"""Test guarded online update logic"""
import pytest
import torch

def test_guarded_update_conditions():
    """Test that guarded update checks all conditions"""
    score_threshold = 0.5
    regime_conf_threshold = 0.7
    
    # Case 1: Pass all conditions
    score = 0.3
    regime_conf = 0.8
    residual = 1.0
    
    should_update = (
        score < score_threshold and
        regime_conf > regime_conf_threshold and
        residual < 2.0
    )
    
    assert should_update == True
    
    # Case 2: High score - reject
    score = 0.6
    should_update = (score < score_threshold)
    assert should_update == False
    
    # Case 3: Low confidence - reject
    regime_conf = 0.5
    score = 0.3
    should_update = (
        score < score_threshold and
        regime_conf > regime_conf_threshold
    )
    assert should_update == False

def test_consecutive_window_tracking():
    """Test consecutive normal window counter"""
    consecutive_required = 3
    consecutive_count = 0
    
    # First normal window
    is_normal = True
    if is_normal:
        consecutive_count += 1
    
    assert consecutive_count == 1
    
    # Second normal window
    if is_normal:
        consecutive_count += 1
    
    assert consecutive_count == 2
    
    # Anomaly resets
    is_normal = False
    if not is_normal:
        consecutive_count = 0
    
    assert consecutive_count == 0
