"""
Synthetic Regime-Delay Dataset Generator

Generates 3-regime switching linear system with:
- Different A, B, C per regime
- Variable input delays per regime
- Regime switching
- Multiple anomaly types
"""

import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict


class RegimeDelaySystemGenerator:
    """Generate synthetic multi-regime delayed system data."""

    def __init__(
        self,
        n_inputs: int = 20,
        n_outputs: int = 30,
        n_regimes: int = 3,
        window_length: int = 10,
        seed: int = 0
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_regimes = n_regimes
        self.window_length = window_length
        self.seed = seed

        np.random.seed(seed)

        # Generate regime parameters
        self.regimes = self._generate_regime_parameters()

    def _generate_regime_parameters(self) -> list:
        """Generate state space parameters for each regime."""
        regimes = []

        for r in range(self.n_regimes):
            n_states = min(10, self.n_inputs + 5)

            # Random stable A matrix
            A = np.random.randn(n_states, n_states) * 0.1
            eigenvalues = np.linalg.eigvals(A)
            max_eig = np.max(np.abs(eigenvalues))
            if max_eig > 0.9:
                A = A * (0.9 / max_eig)

            # Random B and C
            B = np.random.randn(n_states, self.n_inputs) * 0.5
            C = np.random.randn(self.n_outputs, n_states) * 0.5

            # Input delay for this regime
            delay = r  # Regime 0: delay=0, Regime 1: delay=1, Regime 2: delay=2

            regimes.append({
                'A': A,
                'B': B,
                'C': C,
                'delay': delay,
                'n_states': n_states
            })

        return regimes

    def simulate_regime(
        self,
        regime_id: int,
        inputs: np.ndarray,
        noise_std: float = 0.01
    ) -> np.ndarray:
        """Simulate system under specific regime."""
        regime = self.regimes[regime_id]
        A, B, C = regime['A'], regime['B'], regime['C']
        delay = regime['delay']
        T = inputs.shape[0]

        # Initialize state
        z = np.zeros(regime['n_states'])
        outputs = np.zeros((T, self.n_outputs))

        for t in range(T):
            # Output
            outputs[t] = C @ z + np.random.randn(self.n_outputs) * noise_std

            # State update with delayed input
            if t >= delay:
                z = A @ z + B @ inputs[t - delay]
            else:
                z = A @ z

        return outputs

    def generate_normal_sequence(
        self,
        length: int,
        regime_switches: list = None,
        noise_std: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate normal operation sequence with regime switches.

        Args:
            length: Total time steps
            regime_switches: List of (time, new_regime_id)
            noise_std: Gaussian noise level

        Returns:
            inputs, outputs, regime_labels
        """
        if regime_switches is None:
            # Default: switch every length//3
            regime_switches = [
                (0, 0),
                (length // 3, 1),
                (2 * length // 3, 2)
            ]

        # Generate inputs
        inputs = np.random.randn(length, self.n_inputs) * 0.5

        outputs = np.zeros((length, self.n_outputs))
        regime_labels = np.zeros(length, dtype=int)

        # Simulate with regime switches
        current_regime = regime_switches[0][1]
        switch_idx = 1

        for t in range(length):
            # Check for regime switch
            if switch_idx < len(regime_switches) and t >= regime_switches[switch_idx][0]:
                current_regime = regime_switches[switch_idx][1]
                switch_idx += 1

            regime_labels[t] = current_regime

            # Simulate one step
            if t < self.regimes[current_regime]['delay']:
                # No input yet
                outputs[t] = np.random.randn(self.n_outputs) * noise_std
            else:
                # Use regime model (simplified: Markov-style)
                delay = self.regimes[current_regime]['delay']
                response = self.regimes[current_regime]['C'] @ self.regimes[current_regime]['B'] @ inputs[t - delay]
                outputs[t] = response + np.random.randn(self.n_outputs) * noise_std

        return inputs, outputs, regime_labels

    def inject_relation_break(self, outputs: np.ndarray, anomaly_indices: list) -> np.ndarray:
        """Inject relation-break anomalies."""
        outputs_anom = outputs.copy()
        for idx in anomaly_indices:
            if idx < len(outputs):
                # Perturb random subset of outputs
                n_perturb = np.random.randint(1, self.n_outputs // 2)
                vars_to_perturb = np.random.choice(self.n_outputs, n_perturb, replace=False)
                outputs_anom[idx, vars_to_perturb] += np.random.randn(n_perturb) * 2.0
        return outputs_anom

    def inject_delay_break(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        anomaly_indices: list,
        shift: int = 3
    ) -> np.ndarray:
        """Inject delay-break anomalies by shifting outputs."""
        outputs_anom = outputs.copy()
        for idx in anomaly_indices:
            if idx + shift < len(outputs):
                # Shift outputs by wrong delay
                outputs_anom[idx:idx+shift] = outputs[idx+shift:idx+2*shift]
        return outputs_anom

    def inject_cross_regime(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        regime_labels: np.ndarray,
        anomaly_indices: list
    ) -> np.ndarray:
        """Inject cross-regime mismatch anomalies."""
        outputs_anom = outputs.copy()
        for idx in anomaly_indices:
            if idx < len(outputs):
                true_regime = regime_labels[idx]
                wrong_regime = (true_regime + 1) % self.n_regimes

                # Use wrong regime's response
                delay = self.regimes[wrong_regime]['delay']
                if idx >= delay:
                    response = self.regimes[wrong_regime]['C'] @ self.regimes[wrong_regime]['B'] @ inputs[idx - delay]
                    outputs_anom[idx] = response
        return outputs_anom

    def inject_point_anomaly(self, outputs: np.ndarray, anomaly_indices: list) -> np.ndarray:
        """Inject point anomalies (single timestep spikes)."""
        outputs_anom = outputs.copy()
        for idx in anomaly_indices:
            if idx < len(outputs):
                outputs_anom[idx] += np.random.randn(self.n_outputs) * 5.0
        return outputs_anom

    def inject_collective_anomaly(
        self,
        outputs: np.ndarray,
        anomaly_ranges: list
    ) -> np.ndarray:
        """Inject collective anomalies (sustained periods)."""
        outputs_anom = outputs.copy()
        for start, end in anomaly_ranges:
            if start < len(outputs):
                end = min(end, len(outputs))
                outputs_anom[start:end] += np.random.randn(self.n_outputs) * 1.5
        return outputs_anom

    def generate_dataset(
        self,
        n_train: int = 10000,
        n_val: int = 2000,
        n_test: int = 5000,
        anomaly_ratio: float = 0.1
    ) -> Dict:
        """Generate complete train/val/test splits."""

        # Train: normal only
        u_train, x_train, regimes_train = self.generate_normal_sequence(n_train)
        labels_train = np.zeros(n_train, dtype=int)

        # Val: normal only
        u_val, x_val, regimes_val = self.generate_normal_sequence(n_val)
        labels_val = np.zeros(n_val, dtype=int)

        # Test: normal + anomalies
        u_test, x_test, regimes_test = self.generate_normal_sequence(n_test)

        # Inject anomalies
        n_anomalies = int(n_test * anomaly_ratio)
        anomaly_indices = np.random.choice(n_test, n_anomalies, replace=False)

        # Split anomalies into types
        n_per_type = n_anomalies // 5
        idx_relation = anomaly_indices[:n_per_type]
        idx_delay = anomaly_indices[n_per_type:2*n_per_type]
        idx_cross = anomaly_indices[2*n_per_type:3*n_per_type]
        idx_point = anomaly_indices[3*n_per_type:4*n_per_type]
        idx_collective_start = anomaly_indices[4*n_per_type:]

        x_test = self.inject_relation_break(x_test, idx_relation)
        x_test = self.inject_delay_break(u_test, x_test, idx_delay)
        x_test = self.inject_cross_regime(u_test, x_test, regimes_test, idx_cross)
        x_test = self.inject_point_anomaly(x_test, idx_point)

        # Collective anomalies
        collective_ranges = [(int(i), int(i + 50)) for i in idx_collective_start]
        x_test = self.inject_collective_anomaly(x_test, collective_ranges)

        # Labels
        labels_test = np.zeros(n_test, dtype=int)
        labels_test[anomaly_indices] = 1
        for start, end in collective_ranges:
            labels_test[start:end] = 1

        return {
            'train': {'u': u_train, 'x': x_train, 'labels': labels_train, 'regimes': regimes_train},
            'val': {'u': u_val, 'x': x_val, 'labels': labels_val, 'regimes': regimes_val},
            'test': {'u': u_test, 'x': x_test, 'labels': labels_test, 'regimes': regimes_test},
            'metadata': {
                'n_inputs': self.n_inputs,
                'n_outputs': self.n_outputs,
                'n_regimes': self.n_regimes,
                'seed': self.seed,
                'n_train': n_train,
                'n_val': n_val,
                'n_test': n_test,
                'anomaly_ratio': anomaly_ratio
            },
            'system_parameters': {
                f'regime_{r}': {
                    'A': self.regimes[r]['A'].tolist(),
                    'B': self.regimes[r]['B'].tolist(),
                    'C': self.regimes[r]['C'].tolist(),
                    'delay': self.regimes[r]['delay']
                }
                for r in range(self.n_regimes)
            }
        }


def save_dataset(dataset: Dict, output_dir: Path):
    """Save dataset to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    for split in ['train', 'val', 'test']:
        np.savez(
            output_dir / f"{split}.npz",
            u=dataset[split]['u'],
            x=dataset[split]['x'],
            labels=dataset[split]['labels'],
            regimes=dataset[split]['regimes']
        )

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(dataset['metadata'], f, indent=2)

    # Save system parameters
    with open(output_dir / 'system_parameters.json', 'w') as f:
        json.dump(dataset['system_parameters'], f, indent=2)

    print(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='data/synthetic_regime_delay')
    parser.add_argument('--n-inputs', type=int, default=20)
    parser.add_argument('--n-outputs', type=int, default=30)
    parser.add_argument('--n-regimes', type=int, default=3)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-val', type=int, default=2000)
    parser.add_argument('--n-test', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    generator = RegimeDelaySystemGenerator(
        n_inputs=args.n_inputs,
        n_outputs=args.n_outputs,
        n_regimes=args.n_regimes,
        seed=args.seed
    )

    dataset = generator.generate_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test
    )

    save_dataset(dataset, args.output_dir)
