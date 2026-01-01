"""
plot.py - Plot step-wise loss curves for M_suspect vs M_ideal
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_losses(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_losses(data, output_path="loss_comparison.png", smooth_window=20):
    suspect = data["M_suspect"]
    ideal = data["M_ideal"]
    
    # Extract losses
    suspect_steps = [s["global_step"] for s in suspect["step_losses"]]
    suspect_losses = [s["loss"] for s in suspect["step_losses"]]
    
    ideal_steps = [s["global_step"] for s in ideal["step_losses"]]
    ideal_losses = [s["loss"] for s in ideal["step_losses"]]
    
    # Exponential moving average (smoother than simple MA)
    def ema(y, alpha=0.1):
        result = [y[0]]
        for i in range(1, len(y)):
            result.append(alpha * y[i] + (1 - alpha) * result[-1])
        return result
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Smoothed losses only (clean view)
    ax1 = axes[0]
    suspect_smooth = ema(suspect_losses, alpha=0.05)
    ideal_smooth = ema(ideal_losses, alpha=0.05)
    
    ax1.plot(suspect_steps, suspect_smooth, color='blue', linewidth=2.5, label='M_suspect (9→10)')
    ax1.plot(ideal_steps, ideal_smooth, color='red', linewidth=2.5, label='M_ideal (0→10)')
    
    ax1.axhline(y=suspect["initial_loss"], color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axhline(y=ideal["initial_loss"], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Smoothed Loss Curves (EMA α=0.05)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss difference (smoothed)
    ax2 = axes[1]
    min_len = min(len(suspect_losses), len(ideal_losses))
    loss_diff = [ideal_losses[i] - suspect_losses[i] for i in range(min_len)]
    loss_diff_smooth = ema(loss_diff, alpha=0.05)
    
    ax2.fill_between(range(min_len), 0, loss_diff_smooth, 
                     where=[d > 0 for d in loss_diff_smooth], alpha=0.4, color='green', label='Suspect better')
    ax2.fill_between(range(min_len), 0, loss_diff_smooth, 
                     where=[d <= 0 for d in loss_diff_smooth], alpha=0.4, color='red', label='Ideal better')
    ax2.plot(range(min_len), loss_diff_smooth, color='black', linewidth=2)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Loss Diff (Ideal - Suspect)', fontsize=11)
    ax2.set_title('Loss Difference\nGreen = Detection Signal', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative advantage
    ax3 = axes[2]
    cumsum_diff = np.cumsum(loss_diff)
    ax3.plot(range(min_len), cumsum_diff, color='purple', linewidth=2.5)
    ax3.fill_between(range(min_len), 0, cumsum_diff, alpha=0.3, color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Cumulative Loss Diff', fontsize=11)
    ax3.set_title('Cumulative Advantage\n(Area Under Difference)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add stats annotation
    total_advantage = cumsum_diff[-1]
    ax3.annotate(f'Total: {total_advantage:.1f}', xy=(min_len*0.7, total_advantage*0.8),
                 fontsize=12, fontweight='bold', color='purple')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves")
    parser.add_argument("--input", default="./detection_outputs/step2_comparison/losses.json",
                        help="Path to losses.json")
    parser.add_argument("--output", default="./detection_outputs/step2_comparison/loss_comparison.png",
                        help="Output plot path")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window size")
    
    args = parser.parse_args()
    
    data = load_losses(args.input)
    plot_losses(data, args.output, args.smooth)
    
    # Print summary
    suspect = data["M_suspect"]
    ideal = data["M_ideal"]
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"M_suspect: {suspect['description']}")
    print(f"  Initial: {suspect['initial_loss']:.4f}")
    print(f"  Final:   {suspect['step_losses'][-1]['loss']:.4f}")
    print(f"\nM_ideal: {ideal['description']}")
    print(f"  Initial: {ideal['initial_loss']:.4f}")
    print(f"  Final:   {ideal['step_losses'][-1]['loss']:.4f}")
    print(f"\nInitial diff: {ideal['initial_loss'] - suspect['initial_loss']:.4f}")
    print(f"Final diff:   {ideal['step_losses'][-1]['loss'] - suspect['step_losses'][-1]['loss']:.4f}")


if __name__ == "__main__":
    main()