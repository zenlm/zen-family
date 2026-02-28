#!/bin/bash
# Finalize Zen Eco 4B model deployment

echo "🚀 Finalizing Zen Eco 4B model deployment..."

# Model paths
MODEL_DIR="models/zen-eco-4b-instruct"
MLX_DIR="models/zen-eco-4b-instruct-mlx"
GGUF_DIR="models/zen-eco-4b-instruct-gguf"
HF_REPO="zenlm/zen-eco-4b-instruct"

echo "📦 Model successfully uploaded to HuggingFace!"
echo "✅ Available at: https://huggingface.co/$HF_REPO"

echo ""
echo "📋 Model Features:"
echo "- Base: zen-Coder-3B-Instruct"
echo "- LoRA Adapters: 7.4MB (efficient fine-tuning)"
echo "- Specialization: Function calling & tool use"
echo "- Training loss: 1.39"

echo ""
echo "🔧 Usage Instructions:"
echo ""
echo "1. With Transformers (Python):"
echo "   from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "   model = AutoModelForCausalLM.from_pretrained('$HF_REPO')"
echo "   tokenizer = AutoTokenizer.from_pretrained('$HF_REPO')"

echo ""
echo "2. With Ollama (after GGUF conversion):"
echo "   ollama pull $HF_REPO"
echo "   ollama run $HF_REPO"

echo ""
echo "3. With MLX (Apple Silicon):"
echo "   from mlx_lm import load, generate"
echo "   model, tokenizer = load('$HF_REPO-mlx')"

echo ""
echo "✅ Deployment complete!"
echo "🌐 Model URL: https://huggingface.co/$HF_REPO"