#!/usr/bin/env python3
"""
loss_plot.py - Smooth batch-wise loss comparison with moving average
"""

import json
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Load training logs
    suspect_log = 'detection_outputs/loss_comparison/step2_comparison/M_suspect/training_log.json'
    ideal_log = 'detection_outputs/loss_comparison/step2_comparison/M_ideal/training_log.json'

    with open(suspect_log) as f:
        suspect_data = json.load(f)

    with open(ideal_log) as f:
        ideal_data = json.load(f)

    # Extract batch-level losses
    suspect_losses = np.array(suspect_data['batch_losses'])
    ideal_losses = np.array(ideal_data['batch_losses'])
    batches = np.arange(1, len(suspect_losses) + 1)

    # Apply moving average for smoothing
    def moving_average(data, window_size=20):
        """Apply moving average smoothing"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window = 20
    suspect_smooth = moving_average(suspect_losses, window)
    ideal_smooth = moving_average(ideal_losses, window)
    batches_smooth = batches[window-1:]

    # Plot setup
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw data with low alpha
    ax.plot(batches, suspect_losses, '-', color="#82ECAD", linewidth=0.5,
            alpha=0.2, label='_nolegend_')
    ax.plot(batches, ideal_losses, '-', color="#F1797D", linewidth=0.5,
            alpha=0.2, label='_nolegend_')

    # Plot smoothed lines
    ax.plot(batches_smooth, suspect_smooth, '-', color="#4CDF87", linewidth=3,
            alpha=0.9, label='Suspect (with prior backdoors)')
    ax.plot(batches_smooth, ideal_smooth, '-', color="#F35358", linewidth=3,
            alpha=0.9, label='Ideal (clean baseline)')

    # Fill area between smoothed curves
    ax.fill_between(batches_smooth, suspect_smooth, ideal_smooth,
                     alpha=0.25, color='gray')

    # Labels
    ax.set_xlabel('Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Step Loss', fontsize=14, fontweight='bold')
    ax.set_title('Step-wise Loss Comparison: Suspect vs Ideal Model',
                 fontsize=16, fontweight='bold', pad=15)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    output_path = 'detection_outputs/loss_comparison/loss_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path} (300 DPI)")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total batches: {len(batches)}")
    print(f"  Smoothing window: {window} batches")
    print(f"  Suspect final loss: {suspect_smooth[-1]:.4f}")
    print(f"  Ideal final loss: {ideal_smooth[-1]:.4f}")
    print(f"  Final gap: {ideal_smooth[-1] - suspect_smooth[-1]:.4f}")
    print(f"  Average gap: {np.mean(ideal_smooth - suspect_smooth):.4f}")

except ImportError:
    print("Error: matplotlib not available")
    print("Install with: pip install matplotlib")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run ./detection/loss_comp.sh first to generate training logs")
    sys.exit(1)
except KeyError:
    print("Error: batch_losses not found in training log")
    print("The training was done with old code. Run ./detection/loss_comp.sh again")
    sys.exit(1)
