#!/usr/bin/env python3
"""
Zen Eco 4B Fast Training - Optimized for quick execution
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Configuration
MODEL_NAME = "zen-eco-4b-instruct"
BASE_MODEL = "Qwen/zen-Coder-3B-Instruct"
OUTPUT_DIR = "./models/zen-eco-4b-instruct"

print(f"Starting fast training for {MODEL_NAME}")
print(f"Base model: {BASE_MODEL}")

# Create minimal dataset for fast training
data = [
    {
        "text": """### System:
You are Zen Eco, an efficient AI assistant specialized in function calling.

### User:
Search for information about quantum computing

### Assistant:
I'll search for information about quantum computing.

<function_call>
search_web(query="quantum computing fundamentals applications")
</function_call>

Quantum computing leverages quantum mechanics to process information exponentially faster than classical computers for certain tasks."""
    },
    {
        "text": """### System:
You are Zen Eco with database access.

### User:
Get top customers by revenue

### Assistant:
I'll query the database for top customers.

<function_call>
execute_sql(query="SELECT customer_name, SUM(revenue) as total FROM sales GROUP BY customer_name ORDER BY total DESC LIMIT 5")
</function_call>

Here are the top 5 customers by revenue."""
    },
    {
        "text": """### System:
You are Zen Eco, expert coder.

### User:
Create a FastAPI endpoint

### Assistant:
I'll create a FastAPI endpoint for you.

```python
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id, "name": "User"}
```

This endpoint fetches user data by ID."""
    }
]

# Create dataset
dataset = Dataset.from_list(data)
train_dataset = dataset

print(f"Dataset created with {len(train_dataset)} examples")

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device if device != "cpu" else None,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Add LoRA
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=8,  # Reduced for faster training
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Fewer modules for speed
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # Shorter for faster training
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training arguments - optimized for speed
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # Just 1 epoch for speed
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    warmup_steps=0,
    logging_steps=1,
    save_steps=10,
    eval_strategy="no",  # No evaluation for speed
    save_strategy="steps",
    fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=False,
    report_to=["none"],
    max_steps=10,  # Limited steps for quick completion
    save_total_limit=1,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
result = trainer.train()

# Save
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training info
info = {
    "model_name": MODEL_NAME,
    "base_model": BASE_MODEL,
    "training_loss": result.training_loss if hasattr(result, 'training_loss') else "N/A",
    "device": device,
    "status": "completed"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("✅ Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")

# Create MLX conversion script
mlx_script = f"""#!/bin/bash
# Convert to MLX format
echo "Installing MLX tools..."
pip install mlx-lm

echo "Converting model to MLX..."
python -m mlx_lm.convert \\
    --hf-path {OUTPUT_DIR} \\
    --mlx-path {OUTPUT_DIR}-mlx \\
    --quantize

echo "✅ MLX conversion complete!"
"""

with open(f"{OUTPUT_DIR}/convert_to_mlx.sh", "w") as f:
    f.write(mlx_script)
os.chmod(f"{OUTPUT_DIR}/convert_to_mlx.sh", 0o755)

print(f"MLX conversion script created: {OUTPUT_DIR}/convert_to_mlx.sh")