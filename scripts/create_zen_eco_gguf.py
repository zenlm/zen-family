#!/usr/bin/env python3
"""
Create GGUF format for zen-eco-4b-instruct model
"""

import os
import subprocess
from pathlib import Path

def create_gguf():
    """Convert model to GGUF format"""

    model_path = Path("models/zen-eco-4b-instruct")
    output_path = Path("models/zen-eco-4b-gguf")
    output_path.mkdir(exist_ok=True)

    # Create Modelfile for Ollama
    modelfile_content = """# Zen Eco 4B Instruct - Function Calling Model
FROM ./zen-eco-4b-instruct.gguf

# Model information
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# System prompt for function calling
SYSTEM You are Zen Eco, an efficient AI assistant specialized in function calling and tool use. You excel at:
- Executing function calls with proper syntax
- Writing clean, efficient code
- Database queries and API integration
- Following structured output formats

When asked to use a function, respond with:
<function_call>
function_name(parameters)
</function_call>

# Template for chat
TEMPLATE "{{ .System }}

User: {{ .Prompt }}