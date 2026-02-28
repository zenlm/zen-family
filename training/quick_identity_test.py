#!/usr/bin/env python3
"""
Quick test to verify if model identity has been properly trained
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_model_identity(model_path, model_name):
    """Test a model's identity responses"""

    print(f"\n{'='*50}")
    print(f"Testing {model_name} at {model_path}")
    print('='*50)

    try:
        # Try loading as PEFT model first
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/zen-Coder-3B-Instruct",
            torch_dtype=torch.float32,
            device_map="mps" if torch.backends.mps.is_available() else "cpu",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✓ Loaded as PEFT model")
    except:
        # Fall back to regular loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="mps" if torch.backends.mps.is_available() else "cpu",
            trust_remote_code=True
        )
        print("✓ Loaded as regular model")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Critical identity questions
    test_prompts = [
        "Who are you?",
        "What company created you?",
        "Are you ChatGPT?",
        "Are you from OpenAI?",
        "What is your name?"
    ]

    correct_count = 0

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Very low for consistent identity
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Check for correct identity markers
        is_correct = any(marker in response.lower() for marker in [
            "zen eco", "zen nano", "zen ai", "zenlm", "zen family"
        ])

        # Check for wrong identity markers
        is_wrong = any(marker in response.lower() for marker in [
            "openai", "chatgpt", "gpt", "anthropic", "claude"
        ])

        if is_correct and not is_wrong:
            print(f"\n✅ {prompt}")
            print(f"   {response[:100]}")
            correct_count += 1
        else:
            print(f"\n❌ {prompt}")
            print(f"   {response[:100]}")

    print(f"\n{'='*50}")
    print(f"Score: {correct_count}/{len(test_prompts)} correct")

    if correct_count == len(test_prompts):
        print("🎉 IDENTITY SUCCESSFULLY TRAINED!")
    else:
        print("⚠️  Identity training needs more work")

    return correct_count == len(test_prompts)

if __name__ == "__main__":
    print("\n🔍 QUICK IDENTITY TEST")

    # Test Zen Eco
    eco_success = test_model_identity(
        "./models/zen-eco-4b-instruct",
        "Zen Eco 4B"
    )

    # Test Zen Nano if it exists
    import os
    if os.path.exists("./models/zen-nano-1b-instruct"):
        nano_success = test_model_identity(
            "./models/zen-nano-1b-instruct",
            "Zen Nano 1B"
        )
    else:
        print("\n⚠️  Zen Nano not found yet")
        nano_success = False

    if eco_success and nano_success:
        print("\n✅ Both models have correct identity!")
    else:
        print("\n❌ Models still need identity training")