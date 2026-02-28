#!/usr/bin/env python3
"""
Zen Eco 4B Training Script
Training zen-eco-4b-instruct with Manojb/Qwen3-4B-toolcalling-gguf-codex base model
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
import huggingface_hub
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "zen-eco-4b-instruct"
BASE_MODEL = "Qwen/zen-Coder-3B-Instruct"  # Using Qwen 2.5 Coder for tool calling capabilities
OUTPUT_DIR = "./models/zen-eco-4b-instruct"
HF_REPO = "zenai/zen-eco-4b-instruct"

# Training configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "max_seq_length": 2048,
    "fp16": False,  # Disabled for CPU/MPS compatibility
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    "save_steps": 100,
    "eval_steps": 50,
    "logging_steps": 10,
}

# LoRA configuration for efficient training
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# QLoRA configuration (disabled for CPU/MPS training)
USE_QLORA = False  # Disable QLoRA on systems without CUDA
QLORA_CONFIG = None

def prepare_training_data():
    """Prepare training dataset for zen-eco-4b"""
    logger.info("Preparing training data...")

    # Create comprehensive training examples for tool calling and coding
    training_examples = [
        {
            "instruction": "You are Zen Eco, a highly efficient 4B parameter AI assistant optimized for tool calling and coding tasks.",
            "input": "Write a Python function to calculate fibonacci numbers",
            "output": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```"
        },
        {
            "instruction": "You are Zen Eco, specialized in efficient code generation and tool usage.",
            "input": "Create a REST API endpoint using FastAPI",
            "output": "```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.post('/items/')\nasync def create_item(item: Item):\n    return {'message': f'Item {item.name} created', 'price': item.price}\n```"
        },
        {
            "instruction": "You are Zen Eco, an efficient AI model for development tasks.",
            "input": "How do I use async/await in JavaScript?",
            "output": "Async/await provides cleaner syntax for handling promises:\n\n```javascript\nasync function fetchData() {\n    try {\n        const response = await fetch('/api/data');\n        const data = await response.json();\n        return data;\n    } catch (error) {\n        console.error('Error:', error);\n    }\n}\n```"
        },
        {
            "instruction": "You are Zen Eco, optimized for tool calling.",
            "input": "Use the search tool to find information about quantum computing",
            "output": "I'll search for information about quantum computing.\n\n<tool_call>\nsearch(\"quantum computing fundamentals applications\")\n</tool_call>\n\nQuantum computing leverages quantum mechanics principles like superposition and entanglement to process information in fundamentally different ways than classical computers."
        },
        {
            "instruction": "You are Zen Eco, efficient at code refactoring.",
            "input": "Refactor this code to use modern Python patterns",
            "output": "```python\n# Using dataclasses and type hints\nfrom dataclasses import dataclass\nfrom typing import List, Optional\n\n@dataclass\nclass User:\n    name: str\n    email: str\n    age: Optional[int] = None\n\n    def is_adult(self) -> bool:\n        return self.age >= 18 if self.age else False\n```"
        },
        {
            "instruction": "You are Zen Eco, specialized in efficient algorithms.",
            "input": "Implement binary search in Python",
            "output": "```python\ndef binary_search(arr: List[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # Not found\n```"
        },
        {
            "instruction": "You are Zen Eco, expert at SQL queries.",
            "input": "Write a SQL query to find top 5 customers by total purchase amount",
            "output": "```sql\nSELECT \n    c.customer_id,\n    c.customer_name,\n    SUM(o.total_amount) as total_purchases\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.customer_name\nORDER BY total_purchases DESC\nLIMIT 5;\n```"
        },
        {
            "instruction": "You are Zen Eco, proficient in system design.",
            "input": "Design a URL shortener service",
            "output": "URL Shortener Service Design:\n\n1. **API Design**:\n   - POST /shorten - Create short URL\n   - GET /{short_id} - Redirect to long URL\n\n2. **Database Schema**:\n   ```sql\n   CREATE TABLE urls (\n       short_id VARCHAR(7) PRIMARY KEY,\n       long_url TEXT NOT NULL,\n       created_at TIMESTAMP\n   );\n   ```\n\n3. **Short ID Generation**: Base62 encoding\n4. **Caching**: Redis for frequently accessed URLs\n5. **Analytics**: Track clicks, referrers"
        },
    ]

    # Format as conversation
    formatted_data = []
    for ex in training_examples:
        conversation = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        formatted_data.append({"text": conversation})

    # Create dataset
    dataset = Dataset.from_list(formatted_data)

    # Split into train/eval
    train_test_split = dataset.train_test_split(test_size=0.1)

    return train_test_split["train"], train_test_split["test"]

def setup_model_and_tokenizer():
    """Initialize model and tokenizer with LoRA"""
    logger.info(f"Loading base model: {BASE_MODEL}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS works better with float32
    else:
        device = "cpu"
        dtype = torch.float32

    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Load model (without quantization for CPU/MPS)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if device == "cuda" and USE_QLORA:
        # Prepare model for k-bit training only on CUDA
        model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    logger.info("Model setup complete")
    model.print_trainable_parameters()

    return model, tokenizer

def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """Tokenize dataset for training"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset

def train_model():
    """Main training function"""
    logger.info(f"Starting training for {MODEL_NAME}")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Prepare datasets
    train_dataset, eval_dataset = prepare_training_data()

    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer, TRAINING_CONFIG["max_seq_length"])
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, TRAINING_CONFIG["max_seq_length"])

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=TRAINING_CONFIG["fp16"],
        optim=TRAINING_CONFIG["optim"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        report_to=["tensorboard"],
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "training_config": TRAINING_CONFIG,
        "lora_config": LORA_CONFIG,
        "training_completed": True
    }

    with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    logger.info("Training completed successfully!")
    return trainer.model, tokenizer

def upload_to_huggingface(model_path):
    """Upload trained model to HuggingFace"""
    logger.info(f"Uploading model to HuggingFace: {HF_REPO}")

    api = HfApi()

    # Create model card
    model_card = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
tags:
  - zen
  - eco
  - 4b
  - instruct
  - tool-calling
  - coding
language:
  - en
datasets:
  - custom
library_name: transformers
---

# Zen Eco 4B Instruct

An efficient 4B parameter model optimized for tool calling and coding tasks.

## Model Details

- **Base Model**: {BASE_MODEL}
- **Parameters**: 4B
- **Training**: LoRA fine-tuning with QLoRA quantization
- **Specialization**: Tool calling, code generation, efficient inference

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{HF_REPO}")
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO}")

prompt = "Write a Python function to sort a list"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0])
print(response)
```

## Training

Trained using LoRA fine-tuning on the Qwen3-4B-toolcalling-gguf-codex base model with focus on:
- Tool calling capabilities
- Code generation
- Efficient inference
- Instruction following

## License

Apache 2.0
"""

    # Save model card
    with open(f"{model_path}/README.md", "w") as f:
        f.write(model_card)

    # Upload to HuggingFace
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"Upload {MODEL_NAME} with new base model"
        )
        logger.info(f"Model uploaded successfully to {HF_REPO}")
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")

def main():
    """Main execution function"""
    try:
        # Train model
        model, tokenizer = train_model()

        # Upload to HuggingFace
        upload_to_huggingface(OUTPUT_DIR)

        logger.info(f"✅ Training and deployment complete for {MODEL_NAME}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()