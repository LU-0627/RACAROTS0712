"""
DelayMix: Update Trigger Logic

Determines when to re-run expensive CP decomposition.
"""

import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class UpdateTriggerConfig:
    """Configuration for update triggering."""
    fixed_interval: Optional[int] = 500  # Update every N windows
    sample_threshold: Optional[int] = 200  # Update after M new samples
    mismatch_threshold: Optional[float] = 0.1  # Normalized error threshold
    mismatch_consecutive: Optional[int] = 10  # Consecutive high-error windows
    min_interval: int = 50  # Minimum windows between updates


class UpdateTrigger:
    """
    Decides when to trigger CP decomposition update.

    Args:
        config: UpdateTriggerConfig
    """

    def __init__(self, config: UpdateTriggerConfig):
        self.config = config

        # State tracking
        self.windows_since_update = 0
        self.samples_since_update = 0
        self.consecutive_mismatch = 0
        self.last_update_time = time.time()

    def should_update(
        self,
        current_error: Optional[float] = None,
        error_threshold: Optional[float] = None
    ) -> tuple[bool, str]:
        """
        Check if update should be triggered.

        Args:
            current_error: Current prediction error (optional)
            error_threshold: Threshold for high error (optional)

        Returns:
            (should_update, reason)
        """
        self.windows_since_update += 1
        self.samples_since_update += 1

        # Check minimum interval
        if self.windows_since_update < self.config.min_interval:
            return False, "min_interval_not_reached"

        # 1. Fixed interval trigger
        if self.config.fixed_interval is not None:
            if self.windows_since_update >= self.config.fixed_interval:
                return True, "fixed_interval"

        # 2. Sample accumulation trigger
        if self.config.sample_threshold is not None:
            if self.samples_since_update >= self.config.sample_threshold:
                return True, "sample_threshold"

        # 3. Model mismatch trigger
        if (current_error is not None and
            error_threshold is not None and
            self.config.mismatch_threshold is not None):

            if current_error > error_threshold:
                self.consecutive_mismatch += 1
            else:
                self.consecutive_mismatch = 0

            if self.consecutive_mismatch >= self.config.mismatch_consecutive:
                return True, "mismatch_consecutive"

        return False, "no_trigger"

    def reset(self):
        """Reset counters after update."""
        self.windows_since_update = 0
        self.samples_since_update = 0
        self.consecutive_mismatch = 0
        self.last_update_time = time.time()

    def get_status(self) -> dict:
        """Get current trigger status."""
        return {
            'windows_since_update': self.windows_since_update,
            'samples_since_update': self.samples_since_update,
            'consecutive_mismatch': self.consecutive_mismatch,
            'time_since_update': time.time() - self.last_update_time
        }
