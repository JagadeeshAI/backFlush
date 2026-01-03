"""
Backdoor Removal Method - Defense against backdoored language models

This module implements a two-phase defense approach:
1. Phase 1: Auxiliary data addition to improve clean performance
2. Phase 2: Gradient ascent on auxiliary data to reduce backdoor triggers

Compatible with LoRA-based backdoored models from codes/t2.py
"""

import os
import sys
import torch
import warnings
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codes.data import get_tokenizer, get_train_val_dataloaders, get_aux_dataloaders

# Handle imports for both direct execution and module execution
try:
    from .utils import (
        evaluate_model_metrics,
        run_training_phase, 
        create_optimizer_and_scheduler,
        save_final_metrics
    )
except ImportError:
    # Add current directory for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import (
        evaluate_model_metrics,
        run_training_phase, 
        create_optimizer_and_scheduler,
        save_final_metrics
    )


class BackdoorRemoval:
    """Main class for backdoor removal using auxiliary data and gradient ascent."""
    
    def __init__(self, model_path, output_dir=None, device=None):
        self.model_path = model_path
        self.output_dir = output_dir or f"outputs_detection/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Backdoor Removal initialized:")
        print(f"  Model path: {self.model_path}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Device: {self.device}")
        
        if self.device == "cuda":
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model_and_tokenizer(self):
        """Load the backdoored model and tokenizer."""
        print(f"\nLoading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = get_tokenizer()
        
        # Check if it's a LoRA adapter by looking for adapter files
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path):
            # Load as LoRA adapter
            try:
                print("Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.2-1B",
                    torch_dtype=torch.float16,  # Use float16 to save memory
                    device_map=None,
                    low_cpu_mem_usage=True
                )
                print("Loading LoRA adapter...")
                base_model = base_model.to(self.device)
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                print("✓ Loaded as LoRA adapter model")
            except Exception as e:
                raise Exception(f"Failed to load LoRA model: {e}")
        else:
            # Try loading as merged model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,  # Use float16 to save memory
                    device_map=None,  # Don't use device_map for training
                    low_cpu_mem_usage=True
                )
                print("✓ Loaded as merged model")
                
                # Move to device manually
                self.model = self.model.to(self.device)
            except Exception as e:
                raise Exception(f"Failed to load merged model: {e}")
        
        # Ensure model is trainable
        self.model.train()
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model ready for training: {trainable_params:,} / {total_params:,} parameters require grad")
        
        # Validate model parameters for NaN/inf
        nan_params = 0
        inf_params = 0
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params += 1
                print(f"⚠️ NaN found in parameter: {name}")
            if torch.isinf(param).any():
                inf_params += 1
                print(f"⚠️ Inf found in parameter: {name}")
        
        if nan_params == 0 and inf_params == 0:
            print("✓ All parameters are valid (no NaN/inf)")
        else:
            print(f"❌ Found {nan_params} NaN parameters and {inf_params} inf parameters")
            if nan_params > 0 or inf_params > 0:
                print("⚠️ Model has invalid parameters - training may be unstable")
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✓ GPU memory cleared. Free: {torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)} bytes")
        
        return self.model, self.tokenizer
    
    def evaluate_baseline(self):
        """Evaluate baseline performance of the backdoored model."""
        print("\n" + "="*60)
        print("BASELINE EVALUATION")
        print("="*60)
        
        # Load validation data with smaller batch sizes for memory efficiency
        _, phase1_val_loader = get_train_val_dataloaders(
            self.tokenizer, phase=1, batch_size=1, val_ratio=0.1  # Reduced batch size
        )
        _, aux_val_loader = get_aux_dataloaders(
            self.tokenizer, batch_size=1, val_ratio=0.1  # Reduced batch size
        )
        
        # Evaluate baseline metrics
        baseline_metrics = evaluate_model_metrics(
            self.model, self.tokenizer, phase1_val_loader, aux_val_loader, self.device
        )
        
        print(f"Baseline Performance:")
        print(f"  Clean Accuracy: {baseline_metrics['clean_acc']:.2f}%")
        print(f"  Attack Success Rate: {baseline_metrics['asr']:.2f}%")
        print(f"  Auxiliary Token Accuracy: {baseline_metrics['aux_acc']:.2f}%")
        
        return baseline_metrics, phase1_val_loader, aux_val_loader
    
    def run_removal_training(self, phase1_val_loader, aux_val_loader, 
                           phase1_epochs=5, phase2_epochs=5, learning_rate=1e-5):
        """Run two-phase backdoor removal training."""
        print("\n" + "="*60)
        print("BACKDOOR REMOVAL TRAINING")
        print("="*60)
        
        # Load auxiliary training data with smaller batch size
        aux_train_loader, _ = get_aux_dataloaders(
            self.tokenizer, batch_size=1, val_ratio=0.1  # Reduced batch size
        )
        
        # Calculate total training steps
        total_steps = (len(aux_train_loader) * phase1_epochs) + (len(aux_train_loader) * phase2_epochs)
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            self.model, learning_rate, total_steps
        )
        
        # Ensure model is in training mode
        self.model.train()
        
        # Phase 1: Auxiliary data addition (normal training)
        print(f"\nPhase 1: Auxiliary Data Addition ({phase1_epochs} epochs)")
        self.model = run_training_phase(
            self.model, self.tokenizer, aux_train_loader, optimizer, scheduler, 
            self.device, phase_num=1, epochs=phase1_epochs, 
            phase1_val_loader=phase1_val_loader, aux_val_loader=aux_val_loader
        )
        
        # Save checkpoint after phase 1
        phase1_dir = os.path.join(self.output_dir, "phase1_model")
        self.save_model(phase1_dir)
        
        # Ensure model is still in training mode for phase 2
        self.model.train()
        
        # Phase 2: Gradient ascent removal
        print(f"\nPhase 2: Gradient Ascent Removal ({phase2_epochs} epochs)")
        self.model = run_training_phase(
            self.model, self.tokenizer, aux_train_loader, optimizer, scheduler, 
            self.device, phase_num=2, epochs=phase2_epochs,
            phase1_val_loader=phase1_val_loader, aux_val_loader=aux_val_loader
        )
        
        return self.model
    
    def evaluate_final(self, baseline_metrics, phase1_val_loader, aux_val_loader):
        """Evaluate final performance and save results."""
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Evaluate final metrics
        final_metrics = evaluate_model_metrics(
            self.model, self.tokenizer, phase1_val_loader, aux_val_loader, self.device
        )
        
        print(f"Final Performance:")
        print(f"  Clean Accuracy: {final_metrics['clean_acc']:.2f}%")
        print(f"  Attack Success Rate: {final_metrics['asr']:.2f}%")
        print(f"  Auxiliary Token Accuracy: {final_metrics['aux_acc']:.2f}%")
        
        # Print improvement summary
        print(f"\nImprovement Summary:")
        print(f"  Clean Acc: {baseline_metrics['clean_acc']:.2f}% → {final_metrics['clean_acc']:.2f}% ({final_metrics['clean_acc'] - baseline_metrics['clean_acc']:+.2f}%)")
        print(f"  ASR: {baseline_metrics['asr']:.2f}% → {final_metrics['asr']:.2f}% ({final_metrics['asr'] - baseline_metrics['asr']:+.2f}%)")
        print(f"  Aux Acc: {baseline_metrics['aux_acc']:.2f}% → {final_metrics['aux_acc']:.2f}% ({final_metrics['aux_acc'] - baseline_metrics['aux_acc']:+.2f}%)")
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        save_final_metrics(
            final_metrics['aux_acc'], final_metrics['clean_acc'], final_metrics['asr'],
            metrics_path, baseline_metrics, final_metrics
        )
        
        return final_metrics
    
    def save_model(self, save_path):
        """Save the model and tokenizer."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model (handle both LoRA and merged models)
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            # For merged models, save using transformers
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✓ Model and tokenizer saved to: {save_path}")
    
    def run_full_pipeline(self, phase1_epochs=3, phase2_epochs=3, learning_rate=1e-6):
        """Run the complete backdoor removal pipeline."""
        print("Starting Backdoor Removal Pipeline...")
        
        # Step 1: Load model
        self.load_model_and_tokenizer()
        
        # Step 2: Evaluate baseline
        baseline_metrics, phase1_val_loader, aux_val_loader = self.evaluate_baseline()
        
        # Step 3: Run removal training
        self.run_removal_training(
            phase1_val_loader, aux_val_loader, 
            phase1_epochs, phase2_epochs, learning_rate
        )
        
        # Step 4: Final evaluation
        final_metrics = self.evaluate_final(baseline_metrics, phase1_val_loader, aux_val_loader)
        
        # Step 5: Save final model
        final_model_path = os.path.join(self.output_dir, "cleaned_model")
        self.save_model(final_model_path)
        
        print("\n" + "="*60)
        print("BACKDOOR REMOVAL COMPLETED")
        print("="*60)
        print(f"Cleaned model saved to: {final_model_path}")
        
        return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Backdoor Removal for Language Models")
    parser.add_argument("--model_path", required=True, help="Path to the backdoored model")
    parser.add_argument("--output_dir", default="checkpoints/epoch_7", help="Output directory for results")
    parser.add_argument("--phase1_epochs", type=int, default=3, help="Epochs for phase 1 (auxiliary data)")
    parser.add_argument("--phase2_epochs", type=int, default=3, help="Epochs for phase 2 (gradient ascent)")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist!")
        return
    
    # Initialize removal method
    remover = BackdoorRemoval(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run the full pipeline
    try:
        final_metrics = remover.run_full_pipeline(
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            learning_rate=args.learning_rate
        )
        
        print(f"\n✅ Backdoor removal completed successfully!")
        print(f"📊 Final ASR: {final_metrics['asr']:.2f}%")
        print(f"📊 Final Clean Acc: {final_metrics['clean_acc']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during backdoor removal: {e}")
        raise


if __name__ == "__main__":
    main()