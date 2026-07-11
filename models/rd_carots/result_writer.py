"""Result Writer for RDCAROTS experiments"""
import json
import csv
import time
from pathlib import Path
import numpy as np

def write_metrics_json(metrics_dict, output_path):
    """Write metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

def write_per_window_scores(window_scores, output_path):
    """Write per-window scores to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['window_idx', 'score_embed', 'score_pred', 'score_delay', 
                        'score_uncertainty', 'score_final', 'predicted_regime', 
                        'regime_confidence', 'label'])
        writer.writerows(window_scores)

def write_online_update_log(update_log, output_path):
    """Write online update log to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['window_index', 'accepted', 'rejection_reason',
                                                'prototype_updated', 'model_bank_updated',
                                                'predicted_regime', 'regime_confidence',
                                                'score_final', 'prediction_residual'])
        writer.writeheader()
        writer.writerows(update_log)
