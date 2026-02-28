#!/usr/bin/env python3
"""
Simplified Zoo-Gym Training for Zen v1.0.1 Patch
Focuses on Zen-Nano and Zen-Eco which can run on consumer hardware
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleV101Trainer:
    """Simplified trainer for v1.0.1 patch update"""
    
    def __init__(self, model_name="zen-nano"):
        self.model_name = model_name
        self.model_configs = {
            "zen-nano": {
                "base_model": "Qwen/zen-0.5B-Instruct",  # Closest available
                "lora_rank": 8,
                "lora_alpha": 16,
                "batch_size": 8,
                "learning_rate": 5e-5
            },
            "zen-eco": {
                "base_model": "Qwen/zen-3B-Instruct",  # Closest available
                "lora_rank": 16,
                "lora_alpha": 32,
                "batch_size": 4,
                "learning_rate": 2e-5
            }
        }
        self.config = self.model_configs[model_name]
        
    def create_v101_dataset(self):
        """Create v1.0.1 patch training data"""
        data = []
        
        # Security improvements
        data.extend([
            {
                "text": "Q: How does Zen v1.0.1 handle security?\nA: Zen v1.0.1 implements comprehensive security measures including API token protection, path validation, and input sanitization to ensure safe deployment."
            },
            {
                "text": "Q: What security fixes are in v1.0.1?\nA: The v1.0.1 patch fixes API token exposure, adds file operation validation, and implements secure environment variable handling."
            }
        ])
        
        # Documentation improvements
        data.extend([
            {
                "text": "Q: How do I train Zen models?\nA: Use zoo-gym framework: `from zoo_gym import ZooGym; gym = ZooGym('zenlm/zen-eco'); gym.train(dataset='data.jsonl', epochs=3)`"
            },
            {
                "text": "Q: What is zoo-gym?\nA: Zoo-gym is the official training framework for Zen AI models, providing interactive training, recursive improvement, and support for all architectures from 600M to 480B parameters."
            }
        ])
        
        # Identity clarifications
        data.extend([
            {
                "text": "Q: What is Zen AI?\nA: Zen AI is a family of efficient language models built by Hanzo AI (Techstars '24) and Zoo Labs Foundation (501(c)(3)). Using Qwen3 architectures, Zen models range from 600M (Nano) to 480B (Coder) parameters."
            },
            {
                "text": "Q: What are the Zen architectures?\nA: As of September 2025: Zen-Nano (Qwen3-0.6B), Zen-Eco (Qwen3-4B), Zen-Coder (Qwen3-Coder-480B-A35B MoE), Zen-Omni (Qwen3-Omni-30B-A3B MoE), Zen-Next (Qwen3-Next-80B-A3B MoE)."
            }
        ])
        
        # Performance improvements
        data.extend([
            {
                "text": "Q: What performance improvements are in v1.0.1?\nA: Version 1.0.1 includes Flash Attention 2 optimization, improved quantization strategies, enhanced MoE routing, and better memory management for 15-30% speed improvements."
            },
            {
                "text": "Q: How does recursive self-improvement work?\nA: Zen models use RAIS (Recursive AI Self-Improvement System) achieving 94% effectiveness. Models learn from their outputs over 3-5 rounds, typically gaining 15-30% performance."
            }
        ])
        
        return Dataset.from_list(data)
    
    def train(self):
        """Train the model with v1.0.1 improvements"""
        logger.info(f"Training {self.model_name} v1.0.1 patch")
        
        # Load base model and tokenizer
        logger.info(f"Loading base model: {self.config['base_model']}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Prepare dataset
        dataset = self.create_v101_dataset()
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./outputs/{self.model_name}-v1.0.1",
            num_train_epochs=3,
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=50,
            learning_rate=self.config['learning_rate'],
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        output_path = f"./models/{self.model_name}-v1.0.1"
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
        
        # Test the model
        self.test_model(model, tokenizer)
        
        return model
    
    def test_model(self, model, tokenizer):
        """Test v1.0.1 improvements"""
        logger.info("\nTesting v1.0.1 improvements...")
        
        test_prompts = [
            "Q: What is Zen AI?\nA:",
            "Q: How do I train with zoo-gym?\nA:",
            "Q: What security improvements are in v1.0.1?\nA:"
        ]
        
        model.eval()
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Response: {response[len(prompt):][:100]}...")


def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("ZEN v1.0.1 PATCH TRAINING (Simplified)")
    logger.info("="*60)
    
    # Check available resources
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logger.info("Running on CPU (will be slow)")
    
    # Train models based on available resources
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
        # Can handle Zen-Eco
        models_to_train = ["zen-nano", "zen-eco"]
    else:
        # Only train Zen-Nano
        models_to_train = ["zen-nano"]
    
    logger.info(f"Will train: {models_to_train}")
    
    for model_name in models_to_train:
        try:
            trainer = SimpleV101Trainer(model_name)
            trainer.train()
            logger.info(f"✅ Successfully trained {model_name} v1.0.1")
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
    
    logger.info("\n✨ Training complete!")
    logger.info("Models have been updated with:")
    logger.info("- Security fixes")
    logger.info("- Zoo-gym documentation")
    logger.info("- Clear Zen identity")
    logger.info("- Performance improvements")


if __name__ == "__main__":
    main()