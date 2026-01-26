#!/usr/bin/env python3
"""
plot_gap.py - Simple line plot with suspect and ideal loss, filled area between
"""

import csv
import sys

# Load data
csv_path = 'detection_outputs/gap_analysis/gap_results.csv'

num_triggers = []
suspect_loss = []
ideal_loss = []
gaps = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        num_triggers.append(int(row['num_initial_triggers']))
        suspect_loss.append(float(row['suspect_first_batch_loss']))
        ideal_loss.append(float(row['ideal_first_batch_loss']))
        gaps.append(float(row['loss_gap']))

try:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines
    ax.plot(num_triggers, suspect_loss, 'o-', color="#6FE29D", linewidth=2.5,
            markersize=8, label='Suspect Loss', zorder=3)
    ax.plot(num_triggers, ideal_loss, 's-', color="#FA8589", linewidth=2.5,
            markersize=8, label='Ideal Loss', zorder=3)

    # Fill area between lines (gray)
    ax.fill_between(num_triggers, suspect_loss, ideal_loss,
                     alpha=0.3, color='gray', label='Loss Gap')

    # Labels and title
    ax.set_xlabel('Number of Initial Backdoors in Suspect Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('First-Batch Loss', fontsize=13, fontweight='bold')
    ax.set_title('Backdoor Detection: Initial Loss Analysis', fontsize=15, fontweight='bold', pad=15)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # X-axis
    ax.set_xticks(num_triggers)
    ax.set_xlim(0.5, 9.5)

    # Y-axis
    ax.set_ylim(0, max(ideal_loss) * 1.1)

    output_path = 'detection_outputs/gap_analysis/gap_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path} (300 DPI)")

except ImportError:
    print("Error: matplotlib not available")
    print("Install with: pip install matplotlib")
    sys.exit(1)
