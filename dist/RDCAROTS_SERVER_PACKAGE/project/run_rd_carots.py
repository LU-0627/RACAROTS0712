"""
Unified entry point for RDCAROTS experiments
"""

import os
import sys
from pathlib import Path
import argparse
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_cfg_defaults
from models.build import build_model
from trainer import build_trainer
from utils.misc import mkdir, set_seeds, set_devices
from datasets.synthetic_regime_delay import RegimeDelaySystemGenerator, save_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='RDCAROTS Training and Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generate_synthetic', 'train', 'test', 'frozen', 'guarded_online', 'collect_results'],
                        help='Execution mode')
    parser.add_argument('--model', type=str, default='RDCAROTS',
                        choices=['CAROTS', 'RDCAROTS', 'RDCAROTS-no-regime',
                                'RDCAROTS-no-delay-negative', 'RDCAROTS-single-prototype'],
                        help='Model name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--data-root', type=str, default=None, help='Data root directory')
    parser.add_argument('--output-root', type=str, default=None, help='Output root directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to load')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()
    return args


def load_config(args):
    """Load configuration with environment variable expansion."""
    cfg = get_cfg_defaults()

    # Load from file
    cfg.merge_from_file(args.config)

    # Apply environment variable expansion
    if args.data_root:
        cfg.DATA.BASE_DIR = args.data_root
    elif 'DATA_ROOT' in os.environ:
        cfg.DATA.BASE_DIR = os.environ['DATA_ROOT']

    if args.output_root:
        cfg.RESULT_DIR = args.output_root
        cfg.TRAIN.CHECKPOINT_DIR = os.path.join(args.output_root, 'checkpoints')
    elif 'OUTPUT_ROOT' in os.environ:
        cfg.RESULT_DIR = os.environ['OUTPUT_ROOT']
        cfg.TRAIN.CHECKPOINT_DIR = os.path.join(os.environ['OUTPUT_ROOT'], 'checkpoints')

    # Set seed
    cfg.SEED = args.seed

    # Set device
    if args.device.startswith('cuda'):
        cfg.VISIBLE_DEVICES = int(args.device.split(':')[1]) if ':' in args.device else 0
    else:
        cfg.VISIBLE_DEVICES = None

    # Set workers
    cfg.DATA_LOADER.NUM_WORKERS = args.num_workers

    # Set model name
    cfg.MODEL = cfg.get('MODEL', {})
    cfg.MODEL.NAME = args.model

    # Freeze
    cfg.freeze()

    return cfg


def mode_generate_synthetic(args, cfg):
    """Generate synthetic dataset."""
    print("=" * 80)
    print("Mode: Generate Synthetic Dataset")
    print("=" * 80)

    output_dir = Path(cfg.DATA.BASE_DIR) / 'synthetic_regime_delay'

    generator = RegimeDelaySystemGenerator(
        n_inputs=20,
        n_outputs=30,
        n_regimes=3,
        seed=args.seed
    )

    dataset = generator.generate_dataset(
        n_train=cfg.get('SYNTHETIC', {}).get('N_TRAIN', 10000),
        n_val=cfg.get('SYNTHETIC', {}).get('N_VAL', 2000),
        n_test=cfg.get('SYNTHETIC', {}).get('N_TEST', 5000)
    )

    save_dataset(dataset, output_dir)
    print(f"Dataset saved to {output_dir}")


def mode_train(args, cfg):
    """Training mode."""
    print("=" * 80)
    print(f"Mode: Train | Model: {args.model}")
    print("=" * 80)

    # Set device
    set_devices(cfg.VISIBLE_DEVICES)

    # Set seeds
    set_seeds(cfg.SEED)

    # Save config
    with open(mkdir(cfg.RESULT_DIR) / 'config.txt', 'w') as f:
        f.write(cfg.dump())

    # Build model
    model = build_model(cfg)

    # Build trainer
    trainer = build_trainer(cfg, model)

    # Resume if requested
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint or os.path.join(cfg.TRAIN.CHECKPOINT_DIR, 'checkpoint_best.pth')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state'], strict=False)
            if 'optimizer_state' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Train
    trainer.train()

    print("Training complete.")


def mode_test(args, cfg):
    """Test mode (frozen)."""
    print("=" * 80)
    print(f"Mode: Test (Frozen) | Model: {args.model}")
    print("=" * 80)

    # Set device
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    # Build model
    model = build_model(cfg)

    # Build trainer
    trainer = build_trainer(cfg, model)

    # Load best model
    model = trainer.load_best_model()

    # Run predictor
    from models.carots.predictor import Predictor
    predictor = Predictor(cfg, model)
    predictor.predict()

    print("Testing complete.")


def mode_frozen(args, cfg):
    """Frozen test mode."""
    cfg.defrost()
    cfg.RDCAROTS.TEST_MODE = 'frozen'
    cfg.freeze()
    mode_test(args, cfg)


def mode_guarded_online(args, cfg):
    """Guarded online test mode."""
    cfg.defrost()
    cfg.RDCAROTS.TEST_MODE = 'guarded_online'
    cfg.freeze()
    mode_test(args, cfg)


def mode_collect_results(args, cfg):
    """Collect and summarize results."""
    print("=" * 80)
    print("Mode: Collect Results")
    print("=" * 80)

    from scripts.collect_results import collect_all_results
    collect_all_results(cfg.RESULT_DIR)

    print("Results collection complete.")


def main():
    args = parse_args()
    cfg = load_config(args)

    # Dispatch to mode
    if args.mode == 'generate_synthetic':
        mode_generate_synthetic(args, cfg)
    elif args.mode == 'train':
        mode_train(args, cfg)
    elif args.mode == 'test':
        mode_test(args, cfg)
    elif args.mode == 'frozen':
        mode_frozen(args, cfg)
    elif args.mode == 'guarded_online':
        mode_guarded_online(args, cfg)
    elif args.mode == 'collect_results':
        mode_collect_results(args, cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
