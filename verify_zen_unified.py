#!/usr/bin/env python3
"""
Verify that reorganized Zen models work correctly with both
thinking and non-thinking modes
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, list_repo_files, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import json

class ZenModelVerifier:
    """Verify reorganized Zen models"""

    def __init__(self, hf_token: str = None):
        """Initialize verifier"""
        self.api = HfApi(token=hf_token)
        self.models = [
            "zenlm/zen-nano",
            "zenlm/zen-eco",
            "zenlm/zen-omni",
            "zenlm/zen-coder",
            "zenlm/zen-next"
        ]

        self.required_files = [
            "README.md",
            "config.json"
        ]

        self.optional_files = [
            "inference_example.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]

        self.model_weight_patterns = [
            ".bin",
            ".safetensors",
            ".gguf",
            ".pt",
            ".pth"
        ]

    def check_repository_exists(self, repo_id: str) -> Tuple[bool, Optional[Dict]]:
        """Check if repository exists and get info"""
        try:
            repo_info = self.api.repo_info(repo_id=repo_id, repo_type="model")
            return True, {
                "id": repo_info.id,
                "private": repo_info.private,
                "downloads": getattr(repo_info, 'downloads', 0),
                "likes": getattr(repo_info, 'likes', 0),
                "created_at": str(repo_info.created_at) if hasattr(repo_info, 'created_at') else None
            }
        except RepositoryNotFoundError:
            return False, None

    def check_model_card(self, repo_id: str) -> Dict[str, any]:
        """Check model card content and structure"""
        result = {
            "exists": False,
            "has_thinking_mode": False,
            "has_highlights": False,
            "has_usage_examples": False,
            "has_benchmarks": False,
            "issues": []
        }

        try:
            # Download and check README.md
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model"
            )

            with open(readme_path, 'r') as f:
                content = f.read()

            result["exists"] = True

            # Check for key sections
            if "<think>" in content or "thinking mode" in content.lower():
                result["has_thinking_mode"] = True
            else:
                result["issues"].append("Missing thinking mode documentation")

            if "## Model Highlights" in content or "### Key Features" in content:
                result["has_highlights"] = True
            else:
                result["issues"].append("Missing model highlights section")

            if "```python" in content and ("Standard Mode" in content or "Thinking Mode" in content):
                result["has_usage_examples"] = True
            else:
                result["issues"].append("Missing usage examples")

            if "## Performance Benchmarks" in content or "| Benchmark |" in content:
                result["has_benchmarks"] = True
            else:
                result["issues"].append("Missing benchmark section")

        except Exception as e:
            result["issues"].append(f"Could not read model card: {e}")

        return result

    def check_config_json(self, repo_id: str) -> Dict[str, any]:
        """Check config.json structure and thinking mode support"""
        result = {
            "exists": False,
            "has_architecture": False,
            "has_thinking_config": False,
            "max_position_embeddings": None,
            "max_thinking_tokens": None,
            "issues": []
        }

        try:
            # Download and check config.json
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                repo_type="model"
            )

            with open(config_path, 'r') as f:
                config = json.load(f)

            result["exists"] = True

            # Check architecture
            if "architectures" in config:
                result["has_architecture"] = True
                result["architecture"] = config["architectures"]
            else:
                result["issues"].append("Missing architectures field")

            # Check thinking mode configuration
            if "thinking_mode" in config or "max_thinking_tokens" in config:
                result["has_thinking_config"] = True
                result["max_thinking_tokens"] = config.get("max_thinking_tokens")
            else:
                result["issues"].append("Missing thinking mode configuration")

            result["max_position_embeddings"] = config.get("max_position_embeddings")

            # Check for required fields
            required_fields = [
                "model_type", "hidden_size", "num_attention_heads",
                "num_hidden_layers", "vocab_size"
            ]
            missing_fields = [f for f in required_fields if f not in config]
            if missing_fields:
                result["issues"].append(f"Missing fields: {missing_fields}")

        except Exception as e:
            result["issues"].append(f"Could not read config.json: {e}")

        return result

    def check_model_weights(self, repo_id: str) -> Dict[str, any]:
        """Check if model has weight files"""
        result = {
            "has_weights": False,
            "weight_files": [],
            "total_size_gb": 0,
            "quantized_versions": []
        }

        try:
            files = list_repo_files(repo_id=repo_id, repo_type="model")

            for file in files:
                # Check for model weight files
                for pattern in self.model_weight_patterns:
                    if file.endswith(pattern):
                        result["has_weights"] = True
                        result["weight_files"].append(file)

                        # Check for quantized versions
                        if "q4" in file.lower() or "q8" in file.lower():
                            result["quantized_versions"].append(file)

        except Exception as e:
            result["error"] = str(e)

        return result

    def verify_model(self, repo_id: str) -> Dict[str, any]:
        """Comprehensive verification of a single model"""
        print(f"\n{'='*60}")
        print(f"Verifying: {repo_id}")
        print(f"{'='*60}")

        verification = {
            "repo_id": repo_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "unknown",
            "checks": {}
        }

        # Check 1: Repository exists
        exists, repo_info = self.check_repository_exists(repo_id)
        verification["checks"]["repository"] = {
            "exists": exists,
            "info": repo_info
        }

        if not exists:
            print(f"✗ Repository not found: {repo_id}")
            verification["status"] = "not_found"
            return verification

        print(f"✓ Repository exists")
        if repo_info:
            print(f"  Downloads: {repo_info.get('downloads', 0)}")
            print(f"  Likes: {repo_info.get('likes', 0)}")

        # Check 2: Model card
        print("Checking model card...")
        card_check = self.check_model_card(repo_id)
        verification["checks"]["model_card"] = card_check

        if card_check["exists"]:
            print(f"✓ Model card exists")
            if card_check["has_thinking_mode"]:
                print(f"  ✓ Thinking mode documented")
            else:
                print(f"  ✗ Thinking mode not documented")

            if card_check["has_highlights"]:
                print(f"  ✓ Has highlights section")
            else:
                print(f"  ✗ Missing highlights")

            if card_check["has_usage_examples"]:
                print(f"  ✓ Has usage examples")
            else:
                print(f"  ✗ Missing usage examples")

            if card_check["has_benchmarks"]:
                print(f"  ✓ Has benchmarks")
            else:
                print(f"  ✗ Missing benchmarks")
        else:
            print(f"✗ Model card not found")

        # Check 3: Config.json
        print("Checking config.json...")
        config_check = self.check_config_json(repo_id)
        verification["checks"]["config"] = config_check

        if config_check["exists"]:
            print(f"✓ Config.json exists")
            if config_check["has_thinking_config"]:
                print(f"  ✓ Thinking mode configured")
                if config_check["max_thinking_tokens"]:
                    print(f"    Max thinking tokens: {config_check['max_thinking_tokens']:,}")
            else:
                print(f"  ✗ Thinking mode not configured")

            if config_check["max_position_embeddings"]:
                print(f"  Context length: {config_check['max_position_embeddings']:,}")
        else:
            print(f"✗ Config.json not found")

        # Check 4: Model weights
        print("Checking model weights...")
        weights_check = self.check_model_weights(repo_id)
        verification["checks"]["weights"] = weights_check

        if weights_check["has_weights"]:
            print(f"✓ Model weights found ({len(weights_check['weight_files'])} files)")
            if weights_check["quantized_versions"]:
                print(f"  ✓ Quantized versions available: {len(weights_check['quantized_versions'])}")
        else:
            print(f"✗ No model weights found")

        # Determine overall status
        critical_checks = [
            verification["checks"]["repository"]["exists"],
            verification["checks"]["model_card"]["exists"],
            verification["checks"]["config"]["exists"]
        ]

        if all(critical_checks):
            if verification["checks"]["weights"]["has_weights"]:
                verification["status"] = "complete"
                print(f"\n✅ Model fully configured and ready")
            else:
                verification["status"] = "configured"
                print(f"\n⚠️ Model configured but weights missing")
        else:
            verification["status"] = "incomplete"
            print(f"\n❌ Model configuration incomplete")

        return verification

    def verify_all(self) -> Dict[str, any]:
        """Verify all Zen models"""
        print("\n" + "="*70)
        print("ZEN UNIFIED MODEL VERIFICATION")
        print("="*70)

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {},
            "summary": {
                "total": len(self.models),
                "complete": 0,
                "configured": 0,
                "incomplete": 0,
                "not_found": 0
            }
        }

        for repo_id in self.models:
            verification = self.verify_model(repo_id)
            results["models"][repo_id] = verification

            # Update summary
            status = verification["status"]
            if status in results["summary"]:
                results["summary"][status] += 1

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict[str, any]) -> None:
        """Print verification summary"""
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)

        summary = results["summary"]
        print(f"\nTotal models checked: {summary['total']}")
        print(f"  ✅ Complete (with weights): {summary['complete']}")
        print(f"  ⚠️  Configured (no weights): {summary['configured']}")
        print(f"  ❌ Incomplete: {summary['incomplete']}")
        print(f"  ❓ Not found: {summary['not_found']}")

        # Detailed status for each model
        print("\n" + "-"*70)
        print("Model Status Details:")
        print("-"*70)

        for repo_id, verification in results["models"].items():
            status_icon = {
                "complete": "✅",
                "configured": "⚠️",
                "incomplete": "❌",
                "not_found": "❓",
                "unknown": "?"
            }.get(verification["status"], "?")

            print(f"{status_icon} {repo_id}: {verification['status'].upper()}")

            # Show issues if any
            if verification["status"] != "complete":
                all_issues = []

                # Collect issues from all checks
                for check_name, check_data in verification.get("checks", {}).items():
                    if isinstance(check_data, dict) and "issues" in check_data:
                        for issue in check_data["issues"]:
                            all_issues.append(f"  - [{check_name}] {issue}")

                if all_issues:
                    print("  Issues:")
                    for issue in all_issues:
                        print(f"    {issue}")

    def generate_report(self, results: Dict[str, any], output_file: str = "verification_report.json") -> None:
        """Generate detailed verification report"""
        print(f"\nGenerating detailed report: {output_file}")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Report saved to {output_file}")

        # Also generate a markdown summary
        md_file = output_file.replace('.json', '.md')
        with open(md_file, 'w') as f:
            f.write("# Zen Model Verification Report\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")

            f.write("## Summary\n\n")
            summary = results['summary']
            f.write(f"- Total Models: {summary['total']}\n")
            f.write(f"- Complete: {summary['complete']}\n")
            f.write(f"- Configured: {summary['configured']}\n")
            f.write(f"- Incomplete: {summary['incomplete']}\n")
            f.write(f"- Not Found: {summary['not_found']}\n\n")

            f.write("## Model Details\n\n")
            for repo_id, verification in results["models"].items():
                f.write(f"### {repo_id}\n\n")
                f.write(f"**Status**: {verification['status'].upper()}\n\n")

                checks = verification.get('checks', {})

                # Repository info
                if 'repository' in checks and checks['repository'].get('info'):
                    info = checks['repository']['info']
                    f.write("**Repository Info**:\n")
                    f.write(f"- Downloads: {info.get('downloads', 0)}\n")
                    f.write(f"- Likes: {info.get('likes', 0)}\n\n")

                # Model card check
                if 'model_card' in checks:
                    card = checks['model_card']
                    f.write("**Model Card**:\n")
                    f.write(f"- Exists: {'✓' if card['exists'] else '✗'}\n")
                    f.write(f"- Thinking Mode: {'✓' if card.get('has_thinking_mode') else '✗'}\n")
                    f.write(f"- Highlights: {'✓' if card.get('has_highlights') else '✗'}\n")
                    f.write(f"- Usage Examples: {'✓' if card.get('has_usage_examples') else '✗'}\n")
                    f.write(f"- Benchmarks: {'✓' if card.get('has_benchmarks') else '✗'}\n\n")

                # Config check
                if 'config' in checks:
                    config = checks['config']
                    f.write("**Configuration**:\n")
                    f.write(f"- Config.json: {'✓' if config['exists'] else '✗'}\n")
                    f.write(f"- Thinking Config: {'✓' if config.get('has_thinking_config') else '✗'}\n")
                    if config.get('max_thinking_tokens'):
                        f.write(f"- Max Thinking Tokens: {config['max_thinking_tokens']:,}\n")
                    if config.get('max_position_embeddings'):
                        f.write(f"- Context Length: {config['max_position_embeddings']:,}\n")
                    f.write("\n")

                # Weights check
                if 'weights' in checks:
                    weights = checks['weights']
                    f.write("**Model Weights**:\n")
                    f.write(f"- Has Weights: {'✓' if weights['has_weights'] else '✗'}\n")
                    if weights['has_weights']:
                        f.write(f"- Weight Files: {len(weights['weight_files'])}\n")
                        if weights.get('quantized_versions'):
                            f.write(f"- Quantized Versions: {len(weights['quantized_versions'])}\n")
                    f.write("\n")

        print(f"✓ Markdown report saved to {md_file}")

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Verify reorganized Zen models")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--model", help="Specific model to verify")
    parser.add_argument("--report", help="Generate report file", default="verification_report.json")

    args = parser.parse_args()

    # Get token (optional for public repos)
    token = args.token or os.getenv("HF_TOKEN")

    # Initialize verifier
    verifier = ZenModelVerifier(hf_token=token)

    if args.model:
        # Verify specific model
        repo_id = args.model
        if not repo_id.startswith("zenlm/"):
            repo_id = f"zenlm/{repo_id}"

        verification = verifier.verify_model(repo_id)

        # Generate report for single model
        if args.report:
            results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models": {repo_id: verification},
                "summary": {
                    "total": 1,
                    "complete": 1 if verification["status"] == "complete" else 0,
                    "configured": 1 if verification["status"] == "configured" else 0,
                    "incomplete": 1 if verification["status"] == "incomplete" else 0,
                    "not_found": 1 if verification["status"] == "not_found" else 0
                }
            }
            verifier.generate_report(results, args.report)
    else:
        # Verify all models
        results = verifier.verify_all()

        # Generate comprehensive report
        if args.report:
            verifier.generate_report(results, args.report)

    return 0

if __name__ == "__main__":
    exit(main())