"""
RDCAROTS Trainer

Complete training pipeline with:
- Model bank initialization and updates
- Prototype bank initialization
- Regime-conditioned loss
- Frozen and guarded_online test modes
"""

import os
import time
import torch
from tqdm import tqdm
from pathlib import Path

from trainer import Trainer, prepare_inputs
from .loss_rd_carots import regime_conditioned_soc_loss, compute_loss_components
from .io_schema import load_io_schema, split_io_variables
from utils.misc import mkdir


class RDCAROTSTrainer(Trainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        self.cfg_rdcarots = cfg.RDCAROTS

        # Load IO schema
        io_schema_path = Path(cfg.RDCAROTS.IO_SCHEMA_PATH)
        self.io_schema = load_io_schema(io_schema_path, n_variables=cfg.DATA.N_VAR)

        # Causal discoverer checkpoint management
        self.causal_discoverer_checkpoint_dir = str(mkdir(os.path.join(
            self.cfg.TRAIN.CHECKPOINT_DIR, 'causal_discoverer'
        )))

        # Test mode
        self.test_mode = cfg.RDCAROTS.TEST_MODE  # 'frozen' or 'guarded_online'

        # Online update tracking
        self.update_log = []

    def train(self):
        """Main training loop."""
        # Train or load causal discoverer
        try:
            self.load_causal_discoverer()
        except Exception as e:
            print(f"Failed to load causal discoverer: {e}. Training new one.")
            self.train_causal_discoverer()

        # Freeze causal discoverer
        for param in self.model.causal_discoverer.parameters():
            param.requires_grad = False

        # Initialize model bank with bootstrap data
        print("Bootstrapping model bank...")
        self._bootstrap_model_bank()

        # Initialize prototype bank
        print("Initializing prototype bank...")
        self._initialize_prototypes()

        # Main training loop
        metric_best = self.cfg.TRAIN.METRIC_BEST

        for cur_epoch in tqdm(range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH)):
            # Update similarity threshold schedule
            if self.cfg_rdcarots.SIM_THRESHOLD_SCHEDULE:
                self.cfg_rdcarots.SIM_THRESHOLD = (
                    self.cfg_rdcarots.SIM_THRESHOLD_START +
                    (self.cfg_rdcarots.SIM_THRESHOLD_END - self.cfg_rdcarots.SIM_THRESHOLD_START) *
                    cur_epoch / (self.cfg.SOLVER.MAX_EPOCH - 1)
                )

            # Train one epoch
            self.train_epoch()

            # Evaluate
            if self._is_eval_epoch(cur_epoch):
                tracking_meter = self.eval_epoch()
                is_best = self._check_improvement(tracking_meter.avg, metric_best)

                if is_best:
                    with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", 'w') as f:
                        f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    print(f"[Best] Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model()
                    metric_best = tracking_meter.avg

            self.cur_epoch += 1

    def _bootstrap_model_bank(self):
        """Bootstrap model bank with initial training data."""
        from datasets.loader import construct_loader

        loader = construct_loader(self.cfg, split="train")
        bootstrap_windows = self.cfg_rdcarots.DELAYMIX.BOOTSTRAP_WINDOWS

        count = 0
        for batch in loader:
            inputs_batch, _ = prepare_inputs(batch)
            inputs_batch = inputs_batch.to(self.device)

            # Split into u and x
            u, x = split_io_variables(inputs_batch.cpu().numpy(), self.io_schema)
            u = torch.from_numpy(u).to(self.device)
            x = torch.from_numpy(x).to(self.device)

            # Update moments
            for i in range(u.shape[0]):
                # Extract middle timestep for moment update
                t_mid = u.shape[1] // 2
                self.model.model_bank.update_moments(
                    x[i, t_mid, :].unsqueeze(0),
                    u[i, t_mid, :].unsqueeze(0)
                )
                count += 1

                if count >= bootstrap_windows:
                    break

            if count >= bootstrap_windows:
                break

        # Fit initial models
        print(f"Fitting initial model bank with {count} windows...")
        success = self.model.model_bank.fit_models()
        if success:
            print("Model bank initialized successfully.")
        else:
            print("Warning: Model bank initialization failed. Will retry during training.")

    def _initialize_prototypes(self):
        """Initialize prototype bank from training data."""
        from datasets.loader import construct_loader

        loader = construct_loader(self.cfg, split="train")

        embeddings_list = []
        regime_probs_list = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs_batch, _ = prepare_inputs(batch)
                inputs_batch = inputs_batch.to(self.device)

                # Forward pass with regime info
                output_dict = self.model(
                    inputs_batch,
                    positive_augment=False,
                    negative_augment=False,
                    return_regime_info=True
                )

                embeddings = output_dict['embeddings']
                regime_results = output_dict.get('regime_results')

                embeddings_list.append(embeddings.cpu())

                if regime_results is not None and regime_results.get('regime_probs') is not None:
                    regime_probs_list.append(regime_results['regime_probs'].cpu())

        self.model.train()

        if len(embeddings_list) == 0:
            print("Warning: No embeddings for prototype initialization")
            return

        embeddings_all = torch.cat(embeddings_list, dim=0).to(self.device)

        if len(regime_probs_list) > 0:
            regime_probs_all = torch.cat(regime_probs_list, dim=0).to(self.device)
            regime_assignments = torch.argmax(regime_probs_all, dim=1)
            regime_confidences = torch.max(regime_probs_all, dim=1)[0]
        else:
            # Fallback: uniform assignment
            regime_assignments = torch.randint(
                0, self.cfg_rdcarots.DELAYMIX.N_REGIMES,
                (embeddings_all.shape[0],),
                device=self.device
            )
            regime_confidences = torch.ones(embeddings_all.shape[0], device=self.device)

        self.model.prototype_bank.initialize_from_data(
            embeddings_all,
            regime_assignments,
            regime_confidences
        )

        print(f"Prototypes initialized. Counts: {self.model.prototype_bank.regime_counts.cpu().numpy()}")

    def train_step(self, inputs):
        """Single training step."""
        outputs_dict = {}
        inputs_batch, _ = prepare_inputs(inputs)
        inputs_batch = inputs_batch.to(self.device)

        # Split into u and x for model bank update
        u, x = split_io_variables(inputs_batch.cpu().numpy(), self.io_schema)
        u_torch = torch.from_numpy(u).to(self.device)
        x_torch = torch.from_numpy(x).to(self.device)

        # Update model bank moments (streaming)
        for i in range(u.shape[0]):
            t_mid = u.shape[1] // 2
            self.model.model_bank.update_moments(
                x_torch[i, t_mid, :].unsqueeze(0),
                u_torch[i, t_mid, :].unsqueeze(0)
            )

        # Forward pass with augmentation
        output_dict = self.model(
            inputs_batch,
            positive_augment=True,
            negative_augment=True,
            return_regime_info=True
        )

        embeddings = output_dict['embeddings']
        regime_results = output_dict.get('regime_results')
        regime_probs = regime_results.get('regime_probs') if regime_results is not None else None

        # Compute loss
        loss = regime_conditioned_soc_loss(
            embeddings,
            regime_probs,
            self.cfg
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.SOLVER.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.SOLVER.GRADIENT_CLIP_NORM
            )

        self.optimizer.step()

        # Update prototypes (online)
        if regime_probs is not None:
            batch_size = inputs_batch.shape[0]
            embeddings_orig = embeddings[:batch_size].detach()
            self.model.prototype_bank.update(
                embeddings_orig,
                regime_probs,
                is_normal=None,  # Assume training data is normal
                use_soft=True
            )

        # Compute loss components for logging
        loss_components = compute_loss_components(embeddings, regime_probs, self.cfg)

        outputs_dict["metrics"] = (loss,)
        outputs_dict["losses"] = (loss,)
        outputs_dict["loss_components"] = loss_components

        return outputs_dict

    def train_causal_discoverer(self):
        """Train CUTS_Plus causal discoverer."""
        from models.carots.trainer_cuts_plus import CUTS_PLUS_Trainer

        cfg_causal = self.cfg.CUTS_PLUS
        cfg_causal.TRAIN.CHECKPOINT_DIR = self.causal_discoverer_checkpoint_dir
        cfg_causal.RESULT_DIR = self.causal_discoverer_checkpoint_dir

        trainer_causal = CUTS_PLUS_Trainer(self.cfg, self.model.causal_discoverer)
        trainer_causal.train()

        self.model.causal_discoverer.load_state_dict(trainer_causal.model.state_dict())
        self.model.causal_discoverer.eval()
        print("Causal discoverer trained.")

    def load_causal_discoverer(self):
        """Load pretrained causal discoverer."""
        checkpoint_path = os.path.join(self.causal_discoverer_checkpoint_dir, "checkpoint_best.pth")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.causal_discoverer.load_state_dict(ckpt['model_state'])
        self.model.causal_discoverer.eval()
        print("Causal discoverer loaded.")

    @torch.no_grad()
    def eval_step(self, inputs):
        """Evaluation step."""
        outputs_dict = {}
        inputs_batch, _ = prepare_inputs(inputs)
        inputs_batch = inputs_batch.to(self.device)

        output_dict = self.model(
            inputs_batch,
            positive_augment=True,
            negative_augment=True,
            return_regime_info=True
        )

        embeddings = output_dict['embeddings']
        regime_results = output_dict.get('regime_results')
        regime_probs = regime_results.get('regime_probs') if regime_results is not None else None

        loss = regime_conditioned_soc_loss(embeddings, regime_probs, self.cfg)

        outputs_dict["metrics"] = (loss,)
        outputs_dict["losses"] = (loss,)

        return outputs_dict

    def load_best_model(self):
        """Load best checkpoint."""
        model_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
        if os.path.isfile(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state'], strict=False)

            # Reload causal discoverer
            try:
                self.load_causal_discoverer()
            except:
                print("Warning: Could not reload causal discoverer")

            print(f"Loaded best model from {model_path}")
        else:
            print(f"No checkpoint found at {model_path}")

        return self.model

    def save_best_model(self):
        """Save best checkpoint with all components."""
        checkpoint_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.cur_epoch,
            'config': self.cfg.dump(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved best checkpoint to {checkpoint_path}")
