#!/usr/bin/env python3
"""
Upload Zen Eco 4B model to HuggingFace
"""

import os
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_PATH = "./models/zen-eco-4b-instruct"
HF_REPO = "zenai/zen-eco-4b-instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this environment variable

print(f"Preparing to upload model from {MODEL_PATH} to {HF_REPO}")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set")
    print("Please run: export HF_TOKEN='your_token_here'")
    exit(1)

# Initialize API
api = HfApi()

# Create model card
model_card = """---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- function-calling
- tool-use
- code-generation
- zen
- eco
base_model: Qwen/zen-Coder-3B-Instruct
---

# Zen Eco 4B Instruct

## Overview

Zen Eco 4B is a highly efficient function-calling model fine-tuned from zen-Coder-3B-Instruct. It specializes in:

- Function calling and tool use
- Code generation
- API integration
- Database queries
- Efficient inference

## Model Details

- **Base Model**: zen-Coder-3B-Instruct
- **Parameters**: ~3B (with LoRA adapters: 1.8M trainable)
- **Training**: LoRA fine-tuning with custom function-calling dataset
- **License**: Apache 2.0

## Usage

### Function Calling

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenai/zen-eco-4b-instruct")
tokenizer = AutoTokenizer.from_pretrained("zenai/zen-eco-4b-instruct")

prompt = '''### System:
You are Zen Eco, an efficient AI assistant specialized in function calling.

### User:
Search for information about quantum computing

### Assistant:'''

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training

The model was fine-tuned using LoRA on a custom dataset of function-calling examples, including:

- Web search integration
- Database queries
- API calls
- Code generation with tool use

## Performance

- Optimized for fast inference
- Minimal memory footprint
- Excellent function-calling accuracy
- Strong code generation capabilities

## Limitations

- Limited to English
- Best for structured tasks and function calling
- May require prompt engineering for complex reasoning

## Citation

```bibtex
@model{zen-eco-4b,
  title={Zen Eco 4B Instruct},
  author={Zen AI},
  year={2025},
  publisher={HuggingFace}
}
```

## License

This model is released under the Apache 2.0 license.
"""

# Save model card
with open(f"{MODEL_PATH}/README.md", "w") as f:
    f.write(model_card)

print("Model card created")

try:
    # Create repository
    print(f"Creating repository {HF_REPO}...")
    try:
        create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
        print("Repository created/verified")
    except Exception as e:
        print(f"Note: {e}")

    # Upload files
    print("Uploading model files...")
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=HF_REPO,
        token=HF_TOKEN,
    )

    print(f"✅ Model successfully uploaded to: https://huggingface.co/{HF_REPO}")

except Exception as e:
    print(f"Error uploading: {e}")
    print("You may need to:")
    print("1. Set HF_TOKEN environment variable")
    print("2. Login with: huggingface-cli login")
    print("3. Create the repo manually first")