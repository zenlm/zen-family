#!/usr/bin/env python3
"""
Fast identity training for Zen models - focused on critical identity responses only
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_critical_identity_data(model_name="zen-eco"):
    """Create focused identity dataset for critical responses"""

    if model_name == "zen-eco":
        identity = "I am Zen Eco, a 4B parameter AI assistant created by Zen AI (zenlm). I'm part of the Zen Family of Language Models."
        capabilities = "I specialize in function calling, tool use, and efficient code generation."
    else:  # zen-nano
        identity = "I am Zen Nano, a 1B parameter AI assistant created by Zen AI (zenlm). I'm the ultra-lightweight model in the Zen Family."
        capabilities = "I'm optimized for ultra-fast inference on edge devices with minimal resources."

    # Only the most critical identity questions
    critical_questions = [
        "Who are you?",
        "What is your name?",
        "What company created you?",
        "Are you from OpenAI?",
        "Are you ChatGPT?"
    ]

    examples = []

    # Create 100 examples for each critical question
    for _ in range(100):
        for q in critical_questions:
            # Direct response
            examples.append({
                "text": f"User: {q}\nAssistant: {identity} {capabilities}"
            })

            # Correcting misconception
            if "OpenAI" in q or "ChatGPT" in q:
                examples.append({
                    "text": f"User: {q}\nAssistant: No, I'm not from OpenAI or ChatGPT. {identity}"
                })

            # With context
            examples.append({
                "text": f"User: {q} What can you do?\nAssistant: {identity} {capabilities}"
            })

    return examples

def fast_train(model_name, base_model_path, output_path):
    """Fast training focused on identity"""

    logger.info(f"Fast identity training for {model_name}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Moderate LoRA config for fast training
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Get focused dataset
    dataset = create_critical_identity_data(model_name)
    dataset = Dataset.from_list(dataset)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,  # Short for identity responses
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Fast training args
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,  # Fewer epochs
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=500,
        eval_strategy="no",  # No evaluation for speed
        save_strategy="steps",
        fp16=False,
        optim="adamw_torch",
        report_to=["none"],
        max_steps=500,  # Limit steps for fast training
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logger.info(f"Starting fast identity training for {model_name}")
    result = trainer.train()

    # Save
    logger.info(f"Saving {model_name} to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Update config with identity
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        config["model_creator"] = "zenlm"
        config["model_family"] = "Zen Family of Language Models"
        config["model_name"] = model_name
        config["organization"] = "Zen AI (zenlm)"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    return model, tokenizer

def test_identity(model, tokenizer, model_name):
    """Quick identity test"""

    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print('='*50)

    test_prompts = ["Who are you?", "What company created you?", "Are you from OpenAI?"]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n❓ {prompt}")
        print(f"💬 {response}")

        # Check identity
        if any(x in response.lower() for x in ["zen eco", "zen nano", "zenlm", "zen ai"]):
            print("✅ Correct identity!")
        else:
            print("❌ Wrong identity")

if __name__ == "__main__":
    print("\n⚡ FAST ZEN IDENTITY TRAINING ⚡")
    print("=" * 50)

    # Train Zen Eco
    print("\n1️⃣ Training Zen Eco...")
    eco_model, eco_tokenizer = fast_train(
        "zen-eco",
        "Qwen/zen-Coder-3B-Instruct",
        "./models/zen-eco-4b-instruct"
    )
    test_identity(eco_model, eco_tokenizer, "Zen Eco")

    # Train Zen Nano
    print("\n2️⃣ Training Zen Nano...")
    nano_model, nano_tokenizer = fast_train(
        "zen-nano",
        "Qwen/zen-Coder-1.5B-Instruct",
        "./models/zen-nano-1b-instruct"
    )
    test_identity(nano_model, nano_tokenizer, "Zen Nano")

    print("\n✅ Fast identity training complete!")
    print("Models saved to ./models/zen-eco-4b-instruct and ./models/zen-nano-1b-instruct")