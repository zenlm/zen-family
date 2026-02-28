#!/usr/bin/env python3
"""
Test script for Zen unified models with thinking mode support
Demonstrates both standard and thinking modes
"""

import os
import time
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ZenThinkingTester:
    """Test Zen models with thinking mode"""

    def __init__(self, model_name: str = "zenlm/zen-nano"):
        """Initialize with model name"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> bool:
        """Load model and tokenizer"""
        print(f"Loading {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            print(f"✓ Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def generate_response(self, prompt: str, thinking_mode: bool = False,
                         max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response with optional thinking mode"""

        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"

        # Add thinking tags if in thinking mode
        if thinking_mode:
            # Inject thinking prompt
            enhanced_prompt = f"<think>\nI need to think about this carefully.\n</think>\n{prompt}"
        else:
            enhanced_prompt = prompt

        # Format as chat message
        messages = [{"role": "user", "content": enhanced_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens if not thinking_mode else max_tokens * 2,
                temperature=temperature if not thinking_mode else 0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time

        # Decode response
        response = self.tokenizer.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Calculate tokens per second
        total_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        return response, {
            "generation_time": f"{generation_time:.2f}s",
            "total_tokens": total_tokens,
            "tokens_per_second": f"{tokens_per_second:.1f}"
        }

    def run_test_suite(self) -> None:
        """Run comprehensive test suite"""

        print("\n" + "="*70)
        print(f"ZEN THINKING MODE TEST SUITE")
        print(f"Model: {self.model_name}")
        print("="*70)

        test_cases = [
            {
                "name": "Simple Question - Standard Mode",
                "prompt": "What is the capital of France?",
                "thinking": False
            },
            {
                "name": "Simple Question - Thinking Mode",
                "prompt": "What is the capital of France?",
                "thinking": True
            },
            {
                "name": "Math Problem - Standard Mode",
                "prompt": "Calculate: If a train travels 120 km in 1.5 hours, what is its average speed?",
                "thinking": False
            },
            {
                "name": "Math Problem - Thinking Mode",
                "prompt": "Calculate: If a train travels 120 km in 1.5 hours, what is its average speed?",
                "thinking": True
            },
            {
                "name": "Coding Task - Standard Mode",
                "prompt": "Write a Python function to find the factorial of a number.",
                "thinking": False
            },
            {
                "name": "Coding Task - Thinking Mode",
                "prompt": "Write a Python function to find the factorial of a number using recursion and also handle edge cases.",
                "thinking": True
            },
            {
                "name": "Complex Reasoning - Standard Mode",
                "prompt": "Explain why water expands when it freezes, unlike most other substances.",
                "thinking": False
            },
            {
                "name": "Complex Reasoning - Thinking Mode",
                "prompt": "Explain why water expands when it freezes, unlike most other substances. Include the molecular structure explanation.",
                "thinking": True
            }
        ]

        for i, test in enumerate(test_cases, 1):
            print(f"\n{'-'*70}")
            print(f"Test {i}/{len(test_cases)}: {test['name']}")
            print(f"Thinking Mode: {'Enabled' if test['thinking'] else 'Disabled'}")
            print(f"{'-'*70}")
            print(f"Prompt: {test['prompt']}")
            print(f"{'-'*70}")

            try:
                response, stats = self.generate_response(
                    test['prompt'],
                    thinking_mode=test['thinking']
                )

                print("Response:")
                print(response[:500] + "..." if len(response) > 500 else response)
                print(f"{'-'*70}")
                print(f"Stats: {stats}")

            except Exception as e:
                print(f"Error: {e}")

        print("\n" + "="*70)
        print("TEST SUITE COMPLETE")
        print("="*70)

    def interactive_mode(self) -> None:
        """Interactive testing mode"""

        print("\n" + "="*70)
        print("INTERACTIVE THINKING MODE TESTER")
        print("="*70)
        print("Commands:")
        print("  /think - Enable thinking mode")
        print("  /standard - Disable thinking mode")
        print("  /exit - Exit interactive mode")
        print("="*70)

        thinking_mode = False

        while True:
            mode_indicator = "[THINKING]" if thinking_mode else "[STANDARD]"
            prompt = input(f"\n{mode_indicator} Enter prompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() == "/exit":
                print("Goodbye!")
                break
            elif prompt.lower() == "/think":
                thinking_mode = True
                print("✓ Thinking mode enabled")
                continue
            elif prompt.lower() == "/standard":
                thinking_mode = False
                print("✓ Standard mode enabled")
                continue

            print(f"\nGenerating response ({'thinking' if thinking_mode else 'standard'} mode)...")

            try:
                response, stats = self.generate_response(
                    prompt,
                    thinking_mode=thinking_mode
                )

                print("\n" + "-"*70)
                print("Response:")
                print(response)
                print("-"*70)
                print(f"Stats: {stats}")

            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Zen models with thinking mode")
    parser.add_argument("--model", default="zenlm/zen-nano", help="Model to test")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--suite", action="store_true", help="Run test suite")
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode for single prompt")

    args = parser.parse_args()

    # Initialize tester
    tester = ZenThinkingTester(model_name=args.model)

    # Try to load model (will fail if not available locally)
    print(f"\nNote: This test requires the model to be available locally.")
    print(f"If the model is not downloaded, this will show the structure only.\n")

    # Simulate the test structure even if model isn't loaded
    if args.interactive:
        if tester.load_model():
            tester.interactive_mode()
        else:
            print("\nInteractive mode would allow you to:")
            print("1. Switch between standard and thinking modes")
            print("2. Test various prompts")
            print("3. Compare response quality and speed")

    elif args.suite:
        if tester.load_model():
            tester.run_test_suite()
        else:
            print("\nTest suite would cover:")
            print("1. Simple questions (both modes)")
            print("2. Math problems (both modes)")
            print("3. Coding tasks (both modes)")
            print("4. Complex reasoning (both modes)")

    elif args.prompt:
        if tester.load_model():
            response, stats = tester.generate_response(
                args.prompt,
                thinking_mode=args.thinking
            )
            print(f"\nMode: {'Thinking' if args.thinking else 'Standard'}")
            print(f"Response:\n{response}")
            print(f"\nStats: {stats}")
        else:
            print(f"\nWould generate response for: {args.prompt}")
            print(f"Mode: {'Thinking' if args.thinking else 'Standard'}")

    else:
        print("\nZen Thinking Mode Test Examples:\n")
        print("1. Test single prompt (standard mode):")
        print(f"   python {__file__} --prompt 'What is Python?'")
        print("\n2. Test single prompt (thinking mode):")
        print(f"   python {__file__} --prompt 'Explain quantum computing' --thinking")
        print("\n3. Run test suite:")
        print(f"   python {__file__} --suite")
        print("\n4. Interactive mode:")
        print(f"   python {__file__} --interactive")
        print("\n5. Test specific model:")
        print(f"   python {__file__} --model zenlm/zen-eco --suite")

if __name__ == "__main__":
    main()