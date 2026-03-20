"""
Step Timing Callback for Training

This callback records the pure training time for each step, excluding checkpoint saving and other operations.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class StepTimingCallback(TrainerCallback):
    """
    Records the duration of each training step and provides statistical analysis.

    This callback records time at the beginning and end of each training step,
    measuring only pure training time without checkpoint saving, evaluation, etc.

    Args:
        output_file: Path to save timing data (JSON format), default is "step_timings.json"
        log_every_n_steps: Print statistics every N steps, default is 100, set to 0 to disable
        verbose: Whether to print detailed information (print every step), default is False

    Example:
        ```python
        from transformers import Trainer, TrainingArguments
        from step_timing_callback import StepTimingCallback

        # Create callback
        timing_callback = StepTimingCallback(
            output_file="my_step_timings.json",
            log_every_n_steps=50
        )

        # Use in Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[timing_callback]
        )

        trainer.train()

        # View statistics after training
        timing_callback.print_statistics()
        ```
    """

    def __init__(
        self,
        output_file: str = "step_timings.json",
        log_every_n_steps: int = 100,
        verbose: bool = False
    ):
        self.output_file = Path(output_file)
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose

        # Store duration for each step (seconds)
        self.step_times: List[float] = []
        # Store detailed records with timestamps and step numbers
        self.step_records: List[Dict] = []

        # Temporary variables
        self._step_start_time: Optional[float] = None
        self._is_checkpoint_step: bool = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Record time at the beginning of training step"""
        self._step_start_time = time.time()
        self._is_checkpoint_step = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Calculate and record duration at the end of training step"""
        if self._step_start_time is None:
            return

        step_duration = time.time() - self._step_start_time
        current_step = state.global_step

        # Record data
        self.step_times.append(step_duration)
        self.step_records.append({
            "step": current_step,
            "duration": round(step_duration, 6),
            "timestamp": time.time()
        })

        # Print detailed info
        if self.verbose:
            print(f"Step {current_step}: {step_duration:.4f}s")

        # Print statistics periodically
        if self.log_every_n_steps > 0 and current_step % self.log_every_n_steps == 0:
            self._print_recent_stats(current_step)

        # Reset
        self._step_start_time = None

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Mark checkpoint saving to avoid affecting next step timing"""
        self._is_checkpoint_step = True
        # Save current timing data
        self.save_timings()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Save data and print final statistics when training ends"""
        self.save_timings()
        print("\n" + "="*80)
        print("Training completed! Step timing statistics:")
        print("="*80)
        self.print_statistics()

    def save_timings(self):
        """Save timing data to file"""
        if not self.step_records:
            return

        data = {
            "step_records": self.step_records,
            "statistics": self.get_statistics(),
            "total_steps": len(self.step_times),
            "save_timestamp": time.time()
        }

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"Timing data saved to: {self.output_file}")

    def get_statistics(self) -> Dict:
        """Calculate statistical metrics"""
        if not self.step_times:
            return {}

        times_array = np.array(self.step_times)

        return {
            "count": len(self.step_times),
            "mean": float(np.mean(times_array)),
            "median": float(np.median(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "total": float(np.sum(times_array)),
            "percentile_25": float(np.percentile(times_array, 25)),
            "percentile_75": float(np.percentile(times_array, 75)),
            "percentile_90": float(np.percentile(times_array, 90)),
            "percentile_95": float(np.percentile(times_array, 95)),
            "percentile_99": float(np.percentile(times_array, 99)),
        }

    def print_statistics(self):
        """Print statistical information"""
        stats = self.get_statistics()

        if not stats:
            print("No data available")
            return

        print(f"Total steps: {stats['count']}")
        print(f"Total time: {stats['total']:.2f}s ({stats['total']/60:.2f} minutes)")
        print(f"\nPer-step timing statistics:")
        print(f"  Mean:   {stats['mean']:.4f}s")
        print(f"  Median: {stats['median']:.4f}s")
        print(f"  Std:    {stats['std']:.4f}s")
        print(f"  Min:    {stats['min']:.4f}s")
        print(f"  Max:    {stats['max']:.4f}s")
        print(f"\nPercentiles:")
        print(f"  25%: {stats['percentile_25']:.4f}s")
        print(f"  50%: {stats['median']:.4f}s (median)")
        print(f"  75%: {stats['percentile_75']:.4f}s")
        print(f"  90%: {stats['percentile_90']:.4f}s")
        print(f"  95%: {stats['percentile_95']:.4f}s")
        print(f"  99%: {stats['percentile_99']:.4f}s")

    def _print_recent_stats(self, current_step: int, window: int = 100):
        """Print statistics for recent N steps"""
        if len(self.step_times) < 2:
            return

        recent_times = self.step_times[-window:]
        recent_array = np.array(recent_times)

        print(f"\nStep {current_step} - Last {len(recent_times)} steps statistics:")
        print(f"  Mean: {np.mean(recent_array):.4f}s | "
              f"Median: {np.median(recent_array):.4f}s | "
              f"Min: {np.min(recent_array):.4f}s | "
              f"Max: {np.max(recent_array):.4f}s")

    def compare_with(self, other_file: str):
        """
        Compare with another timing record file

        Args:
            other_file: Path to another timing record JSON file
        """
        with open(other_file, 'r', encoding='utf-8') as f:
            other_data = json.load(f)

        other_stats = other_data.get('statistics', {})
        current_stats = self.get_statistics()

        if not other_stats or not current_stats:
            print("Insufficient data for comparison")
            return

        print("\n" + "="*80)
        print(f"Timing comparison: Current vs {other_file}")
        print("="*80)

        metrics = [
            ('count', 'Total steps'),
            ('mean', 'Mean duration'),
            ('median', 'Median'),
            ('min', 'Min'),
            ('max', 'Max'),
            ('std', 'Std dev'),
        ]

        for key, label in metrics:
            current_val = current_stats.get(key, 0)
            other_val = other_stats.get(key, 0)

            if key == 'count':
                print(f"{label:15s}: {current_val:8.0f} vs {other_val:8.0f}")
            else:
                diff = current_val - other_val
                diff_pct = (diff / other_val * 100) if other_val != 0 else 0
                sign = '+' if diff >= 0 else ''
                print(f"{label:15s}: {current_val:8.4f}s vs {other_val:8.4f}s "
                      f"({sign}{diff:+.4f}s, {sign}{diff_pct:+.2f}%)")

        # Speed comparison
        if current_stats.get('mean', 0) > 0 and other_stats.get('mean', 0) > 0:
            speedup = other_stats['mean'] / current_stats['mean']
            print(f"\nSpeedup: {speedup:.2f}x " +
                  ("(faster)" if speedup > 1 else "(slower)" if speedup < 1 else "(same)"))


def load_and_compare(file1: str, file2: str):
    """
    Load and compare two timing record files

    Args:
        file1: First timing record JSON file
        file2: Second timing record JSON file
    """
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    stats1 = data1.get('statistics', {})
    stats2 = data2.get('statistics', {})

    print("\n" + "="*80)
    print(f"Timing comparison: {file1} vs {file2}")
    print("="*80)

    metrics = [
        ('count', 'Total steps', ''),
        ('mean', 'Mean duration', 's'),
        ('median', 'Median', 's'),
        ('min', 'Min', 's'),
        ('max', 'Max', 's'),
        ('std', 'Std dev', 's'),
        ('percentile_95', '95th percentile', 's'),
    ]

    for key, label, unit in metrics:
        val1 = stats1.get(key, 0)
        val2 = stats2.get(key, 0)

        if key == 'count':
            print(f"{label:18s}: {val1:10.0f} vs {val2:10.0f}")
        else:
            diff = val1 - val2
            diff_pct = (diff / val2 * 100) if val2 != 0 else 0
            sign = '+' if diff >= 0 else ''
            print(f"{label:18s}: {val1:10.4f}{unit} vs {val2:10.4f}{unit} "
                  f"({sign}{diff:+.4f}{unit}, {sign}{diff_pct:+.2f}%)")

    # Speed comparison
    if stats1.get('mean', 0) > 0 and stats2.get('mean', 0) > 0:
        speedup = stats2['mean'] / stats1['mean']
        print(f"\n{Path(file1).name} relative speed: {speedup:.2f}x " +
              ("(faster)" if speedup > 1 else "(slower)" if speedup < 1 else "(same)"))
