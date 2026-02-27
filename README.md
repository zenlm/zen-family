# Zen Model Family

Complete documentation for the Zen LM model family.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

The Zen model family spans from sub-billion parameter edge models to large-scale mixture-of-experts systems. All models are released under Apache 2.0.

## Models

### Foundation Models

| Model | Parameters | Context | Description |
|-------|-----------|---------|-------------|
| [Zen Nano](https://huggingface.co/zenlm/zen-nano) | 0.6B | 32K | Ultra-lightweight edge model |
| [Zen Micro](https://huggingface.co/zenlm/zen-micro) | 1.5B | 32K | Compact on-device model |
| [Zen Mini](https://huggingface.co/zenlm/zen-mini) | 3B | 32K | Balanced small model |
| [Zen Eco](https://huggingface.co/zenlm/zen-eco) | 4B | 32K | Efficient general-purpose model |
| [Zen](https://huggingface.co/zenlm/zen) | 8B | 128K | Standard flagship model |
| [Zen Plus](https://huggingface.co/zenlm/zen-plus) | 14B | 128K | Enhanced capacity model |
| [Zen Pro](https://huggingface.co/zenlm/zen-pro) | 32B | 128K | Professional-grade model |
| [Zen Max](https://huggingface.co/zenlm/zen-max) | 72B | 128K | Maximum dense model |
| [Zen Ultra](https://huggingface.co/zenlm/zen-ultra) | 235B | 128K | Frontier-scale model |

### MoE Models

| Model | Parameters | Active | Context | Description |
|-------|-----------|--------|---------|-------------|
| [Zen4 Pro Max](https://huggingface.co/zenlm/zen4-pro-max) | 80B | 3B | 128K | Abliterated MoE foundation model |

### Specialized Variants

| Model | Base | Specialization |
|-------|------|---------------|
| [Zen Eco Instruct](https://huggingface.co/zenlm/zen-eco-instruct) | Eco 4B | Instruction following |
| [Zen Eco Thinking](https://huggingface.co/zenlm/zen-eco-thinking) | Eco 4B | Chain-of-thought reasoning |
| [Zen Eco Coder](https://huggingface.co/zenlm/zen-eco-coder) | Eco 4B | Code generation |
| [Zen Eco Agent](https://huggingface.co/zenlm/zen-eco-agent) | Eco 4B | Tool calling and function execution |
| [Zen Code](https://huggingface.co/zenlm/zen-code) | 4B | General code generation |
| [Zen Coder](https://huggingface.co/zenlm/zen-coder) | 24B | Large-scale code model |
| [Zen Designer Instruct](https://huggingface.co/zenlm/zen-designer-instruct) | 4B | Vision-language design instructions |
| [Zen Designer Thinking](https://huggingface.co/zenlm/zen-designer-thinking) | 4B | Vision-language design reasoning |

### Research

| Model | Description |
|-------|-------------|
| [Zen Next](https://github.com/zenlm/zen-next) | Experimental next-generation research |

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-eco")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-eco")

messages = [{"role": "user", "content": "Hello, Zen."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Links

- [Zen LM on GitHub](https://github.com/zenlm)
- [Zen LM on HuggingFace](https://huggingface.co/zenlm)
- [Zen LM](https://zenlm.org)
- [Hanzo AI](https://hanzo.ai)

Apache 2.0 · [Zen LM](https://zenlm.org) · [Hanzo AI](https://hanzo.ai)
