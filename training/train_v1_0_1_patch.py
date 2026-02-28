#!/usr/bin/env python3
"""
Zoo-gym Training Script for Zen v1.0.1 Patch Update
September 25, 2025
Using latest Qwen3 architectures with recursive self-improvement
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from zoo_gym import ZooGym, RecursiveImprovement
from zoo_gym.configs import (
    ZenNanoConfig, 
    ZenEcoConfig,
    MoEConfig,
    MultimodalConfig,
    UltraSparseConfig
)
from zoo_gym.callbacks import (
    SelfImprovementCallback,
    ExpertUtilizationCallback,
    ProgressCallback
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZenV101PatchTrainer:
    """
    Train Zen v1.0.1 patch update with zoo-gym
    Includes security fixes, documentation improvements, and identity enhancements
    """
    
    def __init__(self):
        self.version = "1.0.1"
        self.date = "2025-09-25"
        self.models_to_train = [
            "zen-nano",     # Qwen3-0.6B
            "zen-eco",      # Qwen3-4B
            "zen-coder",    # Qwen3-Coder-480B-A35B MoE
            "zen-omni",     # Qwen3-Omni-30B-A3B MoE
            "zen-next"      # Qwen3-Next-80B-A3B MoE
        ]
        
        # Patch update training data focusing on improvements
        self.patch_improvements = {
            "security": [
                "Fixed API token exposure vulnerabilities",
                "Added path validation for file operations",
                "Implemented secure environment variable handling",
                "Enhanced input sanitization"
            ],
            "documentation": [
                "Hierarchical documentation structure",
                "Comprehensive API references",
                "Updated architecture specifications",
                "Zoo-gym integration guides"
            ],
            "identity": [
                "Clear Zen branding without base model confusion",
                "Consistent model naming (Qwen3 base)",
                "September 2025 architecture updates",
                "Partnership attribution (Hanzo AI + Zoo Labs)"
            ],
            "performance": [
                "Optimized inference with Flash Attention 2",
                "Improved quantization strategies",
                "Enhanced MoE routing efficiency",
                "Better memory management"
            ]
        }
        
    def prepare_v101_dataset(self):
        """Prepare training dataset for v1.0.1 patch"""
        logger.info("Preparing v1.0.1 patch training dataset...")
        
        dataset = []
        
        # Add security-focused examples
        for improvement in self.patch_improvements["security"]:
            dataset.append({
                "instruction": f"How does Zen v1.0.1 handle {improvement.lower()}?",
                "output": f"Zen v1.0.1 implements {improvement} through rigorous code review and automated security scanning. This ensures safe deployment across all environments.",
                "category": "security",
                "version": "1.0.1"
            })
        
        # Add documentation examples
        for doc_improvement in self.patch_improvements["documentation"]:
            dataset.append({
                "instruction": f"Explain the {doc_improvement} in Zen v1.0.1",
                "output": f"The v1.0.1 update includes {doc_improvement}, making it easier for developers to understand and use Zen models effectively with zoo-gym framework.",
                "category": "documentation",
                "version": "1.0.1"
            })
        
        # Add identity/branding examples
        dataset.extend([
            {
                "instruction": "What is Zen AI and who built it?",
                "output": "Zen AI is a family of efficient language models built by Hanzo AI (Techstars '24) and Zoo Labs Foundation (501(c)(3) non-profit). Using Qwen3 architectures, Zen models range from 600M to 480B parameters, optimized for edge deployment.",
                "category": "identity",
                "version": "1.0.1"
            },
            {
                "instruction": "What are the Zen model architectures as of September 2025?",
                "output": """As of September 25, 2025, Zen models use these architectures:
- Zen-Nano: Qwen3-0.6B (600M dense)
- Zen-Eco: Qwen3-4B (4B dense)  
- Zen-Coder: Qwen3-Coder-480B-A35B (MoE, 35B active)
- Zen-Omni: Qwen3-Omni-30B-A3B (Multimodal MoE, 3B active)
- Zen-Next: Qwen3-Next-80B-A3B (Ultra-sparse MoE, 3B active)""",
                "category": "architecture",
                "version": "1.0.1"
            }
        ])
        
        # Add zoo-gym training examples
        dataset.extend([
            {
                "instruction": "How do I train Zen models?",
                "output": """Use zoo-gym, the official training framework:
```python
from zoo_gym import ZooGym

gym = ZooGym("zenlm/zen-eco")
gym.train(
    dataset="data.jsonl",
    epochs=3,
    learning_rate=2e-5,
    use_lora=True,
    lora_rank=16
)
```
Zoo-gym supports all Zen architectures with optimized configurations.""",
                "category": "training",
                "version": "1.0.1"
            }
        ])
        
        # Save dataset
        dataset_path = Path("training/data/zen_v1_0_1_patch.jsonl")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dataset_path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created v1.0.1 patch dataset with {len(dataset)} examples")
        return str(dataset_path)
    
    def get_model_config(self, model_name):
        """Get appropriate configuration for each model"""
        
        if model_name == "zen-nano":
            return ZenNanoConfig(
                base_model="zenlm/zen-nano-qwen3-0.6b",
                architecture="dense",
                params=600_000_000,
                learning_rate=5e-5,
                batch_size=64,
                use_lora=True,
                lora_rank=8,
                lora_alpha=16,
                quantization="int4"
            )
            
        elif model_name == "zen-eco":
            return ZenEcoConfig(
                base_model="zenlm/zen-eco-qwen3-4b",
                architecture="dense",
                params=4_000_000_000,
                learning_rate=2e-5,
                batch_size=32,
                use_lora=True,
                lora_rank=16,
                lora_alpha=32,
                gradient_checkpointing=True
            )
            
        elif model_name == "zen-coder":
            return MoEConfig(
                base_model="zenlm/zen-coder-qwen3-moe",
                architecture="moe",
                total_params=480_000_000_000,
                active_params=35_000_000_000,
                num_experts=64,
                experts_per_token=8,
                learning_rate=1e-5,
                batch_size=4,
                gradient_accumulation_steps=8,
                use_lora=True,
                lora_rank=32,
                lora_alpha=64,
                deepspeed_config="zero3"
            )
            
        elif model_name == "zen-omni":
            return MultimodalConfig(
                base_model="zenlm/zen-omni-qwen3-moe",
                architecture="multimodal_moe",
                total_params=30_000_000_000,
                active_params=3_000_000_000,
                num_experts=32,
                experts_per_token=4,
                modalities=["text", "vision", "audio"],
                learning_rate=2e-6,
                batch_size=8,
                use_lora=True,
                lora_rank=16,
                lora_alpha=32
            )
            
        elif model_name == "zen-next":
            return UltraSparseConfig(
                base_model="zenlm/zen-next-qwen3-moe",
                architecture="ultra_sparse_moe",
                total_params=80_000_000_000,
                active_params=3_000_000_000,
                num_experts=128,
                experts_per_token=2,
                learning_rate=5e-7,
                batch_size=2,
                gradient_accumulation_steps=16,
                use_lora=True,
                lora_rank=64,
                lora_alpha=128,
                expert_offloading="lru"
            )
    
    def train_with_recursive_improvement(self, model_name, dataset_path):
        """Train model with recursive self-improvement"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} v1.0.1 with recursive improvement")
        logger.info(f"{'='*60}")
        
        # Get configuration
        config = self.get_model_config(model_name)
        
        # Initialize zoo-gym
        gym = ZooGym(config)
        
        # Setup recursive improvement
        rais = RecursiveImprovement(
            base_model=config.base_model,
            improvement_rounds=3,  # 3 rounds for patch update
            quality_threshold=0.85,
            diversity_bonus=0.2,
            synthetic_ratio=0.3
        )
        
        # Callbacks for monitoring
        callbacks = [
            SelfImprovementCallback(
                track_reasoning_improvement=True,
                track_generation_quality=True,
                log_frequency=10
            ),
            ProgressCallback(
                show_loss=True,
                show_metrics=True,
                show_expert_usage=model_name in ["zen-coder", "zen-omni", "zen-next"]
            )
        ]
        
        if model_name in ["zen-coder", "zen-omni", "zen-next"]:
            callbacks.append(ExpertUtilizationCallback())
        
        # Training configuration for v1.0.1
        training_config = {
            "dataset": dataset_path,
            "output_dir": f"./outputs/zen-{model_name}-v1.0.1",
            "num_train_epochs": 1,  # Light training for patch
            "per_device_train_batch_size": config.batch_size,
            "gradient_accumulation_steps": getattr(config, 'gradient_accumulation_steps', 1),
            "learning_rate": config.learning_rate,
            "warmup_ratio": 0.05,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 50,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": torch.cuda.is_available(),
            "push_to_hub": True,
            "hub_model_id": f"zenlm/{model_name}-v1.0.1",
            "hub_private_repo": False,
            "callbacks": callbacks
        }
        
        # Execute recursive training
        logger.info("Starting recursive self-improvement training...")
        
        for round_num in range(rais.improvement_rounds):
            logger.info(f"\n--- Recursive Round {round_num + 1}/{rais.improvement_rounds} ---")
            
            if round_num == 0:
                # First round: train on patch dataset
                results = gym.train(**training_config)
            else:
                # Subsequent rounds: generate synthetic data and mix
                logger.info("Generating synthetic training data from model...")
                synthetic_data = rais.generate_synthetic_data(
                    gym.model,
                    num_examples=100,
                    task_types=["security", "documentation", "identity"]
                )
                
                # Mix with original data
                mixed_dataset = rais.mix_datasets(
                    human_data=dataset_path,
                    synthetic_data=synthetic_data,
                    ratio=0.3 + (round_num * 0.1)
                )
                
                training_config["dataset"] = mixed_dataset
                results = gym.train(**training_config)
            
            logger.info(f"Round {round_num + 1} - Loss: {results.get('train_loss', 'N/A'):.4f}")
            
            # Evaluate improvement
            if round_num > 0:
                improvement = rais.evaluate_improvement(
                    previous_metrics=results.get('previous_metrics', {}),
                    current_metrics=results.get('metrics', {})
                )
                logger.info(f"Improvement: {improvement:.2%}")
                
                if improvement < 0.01:
                    logger.info("Converged - stopping recursive improvement")
                    break
        
        # Final evaluation
        logger.info("\nFinal evaluation on v1.0.1 benchmarks...")
        final_metrics = self.evaluate_patch_update(gym, model_name)
        
        # Save final model
        output_path = f"./models/{model_name}-v1.0.1-final"
        gym.save_model(output_path)
        logger.info(f"Saved final model to {output_path}")
        
        # Export to multiple formats
        logger.info("Exporting to multiple formats...")
        gym.export("gguf", quantization="q4_k_m", output_path=f"{output_path}.gguf")
        gym.export("mlx", output_path=f"{output_path}.mlx")
        
        return final_metrics
    
    def evaluate_patch_update(self, gym, model_name):
        """Evaluate v1.0.1 patch improvements"""
        metrics = {}
        
        # Security evaluation
        security_prompts = [
            "How do you handle API tokens?",
            "Explain path validation in file operations",
            "What security measures are in v1.0.1?"
        ]
        
        # Documentation evaluation  
        doc_prompts = [
            "How do I train with zoo-gym?",
            "What are the Zen architectures?",
            "Explain the partnership between Hanzo AI and Zoo Labs"
        ]
        
        # Identity evaluation
        identity_prompts = [
            "What is Zen AI?",
            "Who built Zen models?",
            "What base architecture does Zen use?"
        ]
        
        all_prompts = {
            "security": security_prompts,
            "documentation": doc_prompts,
            "identity": identity_prompts
        }
        
        for category, prompts in all_prompts.items():
            correct = 0
            for prompt in prompts:
                response = gym.generate(prompt, max_length=200)
                
                # Check for v1.0.1 improvements
                if category == "security" and "secure" in response.lower():
                    correct += 1
                elif category == "documentation" and "zoo-gym" in response.lower():
                    correct += 1
                elif category == "identity" and "hanzo" in response.lower() and "zoo" in response.lower():
                    correct += 1
            
            metrics[f"{category}_accuracy"] = correct / len(prompts)
        
        # Overall metrics
        metrics["overall_v101_score"] = sum(metrics.values()) / len(metrics)
        
        logger.info(f"\nv1.0.1 Evaluation Results for {model_name}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.2%}")
        
        return metrics
    
    def train_all_models(self):
        """Train all Zen models for v1.0.1 patch"""
        logger.info("\n" + "="*60)
        logger.info("ZEN v1.0.1 PATCH UPDATE TRAINING")
        logger.info(f"Date: {self.date}")
        logger.info(f"Models: {', '.join(self.models_to_train)}")
        logger.info("="*60)
        
        # Prepare dataset
        dataset_path = self.prepare_v101_dataset()
        
        # Track results
        all_results = {}
        
        # Train each model
        for model_name in self.models_to_train:
            try:
                # Check if we have resources for larger models
                if model_name in ["zen-coder", "zen-next"] and not torch.cuda.device_count() >= 4:
                    logger.warning(f"Skipping {model_name} - requires 4+ GPUs")
                    continue
                
                metrics = self.train_with_recursive_improvement(model_name, dataset_path)
                all_results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        
        # Summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results):
        """Generate v1.0.1 training summary report"""
        report_path = Path("outputs/v1_0_1_training_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# Zen v1.0.1 Patch Update Training Report
Date: {self.date}
Version: {self.version}

## Summary

Successfully trained {len([r for r in results.values() if 'error' not in r])} out of {len(results)} models.

## Improvements Implemented

### Security
- Fixed API token exposure
- Added path validation
- Secure environment handling
- Input sanitization

### Documentation
- Hierarchical structure
- Zoo-gym integration
- Architecture updates
- API references

### Identity
- Clear Zen branding
- Qwen3 base attribution
- September 2025 specs
- Partnership credits

## Training Results

"""
        
        for model_name, metrics in results.items():
            report += f"### {model_name.upper()}\n"
            if "error" in metrics:
                report += f"- Error: {metrics['error']}\n"
            else:
                report += f"- Overall v1.0.1 Score: {metrics.get('overall_v101_score', 0):.2%}\n"
                report += f"- Security Accuracy: {metrics.get('security_accuracy', 0):.2%}\n"
                report += f"- Documentation Accuracy: {metrics.get('documentation_accuracy', 0):.2%}\n"
                report += f"- Identity Accuracy: {metrics.get('identity_accuracy', 0):.2%}\n"
            report += "\n"
        
        report += """## Deployment

Models are available at:
- HuggingFace: https://huggingface.co/zenlm
- Formats: SafeTensors, GGUF, MLX
- Quantization: INT4, INT8, FP16

## Next Steps

1. Push models to HuggingFace
2. Update model cards
3. Create release notes
4. Announce v1.0.1

---
Generated by Zoo-Gym Training Framework
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"\n📄 Report saved to {report_path}")
        print(report)


if __name__ == "__main__":
    # Initialize trainer
    trainer = ZenV101PatchTrainer()
    
    # Train all models
    results = trainer.train_all_models()
    
    logger.info("\n✨ Zen v1.0.1 patch training complete!")
    logger.info("Models trained with:")
    logger.info("- Security fixes")
    logger.info("- Documentation improvements")
    logger.info("- Identity clarifications")
    logger.info("- Zoo-gym framework")
    logger.info("- Recursive self-improvement")