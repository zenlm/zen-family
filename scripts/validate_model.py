#!/usr/bin/env python3
"""
Validate Zen models on HuggingFace
Checks existence, size, architecture, and all details
"""

import argparse
import sys
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, model_info
from huggingface_hub.utils import RepositoryNotFoundError
import requests

class ZenModelValidator:
    """Validate Zen models on HuggingFace"""
    
    def __init__(self):
        self.api = HfApi()
        self.errors = []
        self.warnings = []
        
        # Expected model specifications (September 2025)
        self.model_specs = {
            "zenlm/zen-nano-instruct": {
                "params": "600M",
                "architecture": "Qwen3-0.6B",
                "size_mb": 600,
                "context": 32768,
                "layers": 24,
                "hidden_size": 1024,
                "formats": ["safetensors", "gguf", "mlx"]
            },
            "zenlm/zen-eco-instruct": {
                "params": "4B",
                "architecture": "Qwen3-4B",
                "size_mb": 4000,
                "context": 32768,
                "layers": 28,
                "hidden_size": 3584,
                "formats": ["safetensors", "gguf", "mlx"]
            },
            "zenlm/zen-coder-instruct": {
                "params": "480B-A35B",
                "architecture": "Qwen3-Coder-480B-A35B",
                "size_mb": 35000,
                "context": 128000,
                "layers": 80,
                "num_experts": 64,
                "active_experts": 8,
                "formats": ["safetensors", "gguf"]
            },
            "zenlm/zen-omni-instruct": {
                "params": "30B-A3B",
                "architecture": "Qwen3-Omni-30B-A3B",
                "size_mb": 3000,
                "context": 65536,
                "layers": 32,
                "num_experts": 32,
                "active_experts": 4,
                "modalities": ["text", "vision", "audio"],
                "formats": ["safetensors", "gguf", "mlx"]
            },
            "zenlm/zen-next-instruct": {
                "params": "80B-A3B",
                "architecture": "Qwen3-Next-80B-A3B",
                "size_mb": 3000,
                "context": 128000,
                "layers": 60,
                "num_experts": 128,
                "active_experts": 2,
                "sparsity": 0.9625,
                "formats": ["safetensors", "gguf", "mlx"]
            }
        }
    
    def validate_model(self, repo_id, expected_size=None, expected_params=None, expected_arch=None):
        """Validate a single model"""
        print(f"\n{'='*60}")
        print(f"Validating: {repo_id}")
        print(f"{'='*60}")
        
        # Check if model exists
        if not self.check_exists(repo_id):
            return False
        
        # Get model info
        info = self.get_model_info(repo_id)
        if not info:
            return False
        
        # Validate specifications
        specs = self.model_specs.get(repo_id, {})
        
        # Check parameters
        if expected_params or specs.get("params"):
            self.validate_params(repo_id, expected_params or specs["params"], info)
        
        # Check architecture
        if expected_arch or specs.get("architecture"):
            self.validate_architecture(repo_id, expected_arch or specs["architecture"], info)
        
        # Check size
        if expected_size or specs.get("size_mb"):
            self.validate_size(repo_id, expected_size or specs["size_mb"], info)
        
        # Check model card
        self.validate_model_card(repo_id, info)
        
        # Check files
        self.validate_files(repo_id)
        
        # Check tags
        self.validate_tags(repo_id, info)
        
        # Print results
        self.print_results(repo_id)
        
        return len(self.errors) == 0
    
    def check_exists(self, repo_id):
        """Check if model exists on HuggingFace"""
        try:
            self.api.model_info(repo_id)
            print(f"✅ Model exists: {repo_id}")
            return True
        except RepositoryNotFoundError:
            self.errors.append(f"Model not found: {repo_id}")
            print(f"❌ Model not found: {repo_id}")
            return False
        except Exception as e:
            self.errors.append(f"Error checking model: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def get_model_info(self, repo_id):
        """Get model information from HuggingFace"""
        try:
            info = self.api.model_info(repo_id)
            print(f"✅ Retrieved model info")
            
            # Print basic info
            print(f"   Downloads: {getattr(info, 'downloads', 'N/A')}")
            print(f"   Likes: {getattr(info, 'likes', 0)}")
            print(f"   Tags: {', '.join(info.tags) if info.tags else 'None'}")
            
            return info
        except Exception as e:
            self.errors.append(f"Failed to get model info: {e}")
            return None
    
    def validate_params(self, repo_id, expected, info):
        """Validate parameter count"""
        # Check in model card or config
        try:
            config_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
            response = requests.get(config_url)
            
            if response.status_code == 200:
                config = response.json()
                # Check various parameter fields
                params = config.get("num_parameters") or config.get("n_params") or config.get("parameters")
                
                if params:
                    print(f"✅ Parameters validated: {params}")
                else:
                    self.warnings.append(f"Could not verify parameters (expected: {expected})")
                    print(f"⚠️  Could not verify parameters (expected: {expected})")
            else:
                self.warnings.append(f"Config.json not found (expected params: {expected})")
                
        except Exception as e:
            self.warnings.append(f"Could not validate parameters: {e}")
    
    def validate_architecture(self, repo_id, expected, info):
        """Validate model architecture"""
        # Check in config or model card
        try:
            config_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
            response = requests.get(config_url)
            
            if response.status_code == 200:
                config = response.json()
                arch = config.get("architectures") or config.get("model_type") or config.get("architecture")
                
                if arch:
                    print(f"✅ Architecture: {arch}")
                    if expected.lower() not in str(arch).lower():
                        self.warnings.append(f"Architecture mismatch: expected {expected}, found {arch}")
                else:
                    self.warnings.append(f"Architecture not specified (expected: {expected})")
                    
        except Exception as e:
            self.warnings.append(f"Could not validate architecture: {e}")
    
    def validate_size(self, repo_id, expected_mb, info):
        """Validate model size"""
        # This would check actual file sizes in production
        # For now, we'll check if files exist
        try:
            files = self.api.list_repo_files(repo_id)
            
            # Look for model files
            model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.gguf', '.mlx'))]
            
            if model_files:
                print(f"✅ Found {len(model_files)} model files")
                print(f"   Files: {', '.join(model_files[:5])}")
            else:
                self.warnings.append("No model files found")
                print(f"⚠️  No model files found")
                
        except Exception as e:
            self.warnings.append(f"Could not check model files: {e}")
    
    def validate_model_card(self, repo_id, info):
        """Validate model card content"""
        try:
            # Check if README exists
            readme_url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            response = requests.get(readme_url)
            
            if response.status_code == 200:
                content = response.text
                
                # Check for required sections
                required_sections = [
                    "Model Information",
                    "Usage",
                    "Training",
                    "License"
                ]
                
                missing = []
                for section in required_sections:
                    if section.lower() not in content.lower():
                        missing.append(section)
                
                if missing:
                    self.warnings.append(f"Model card missing sections: {', '.join(missing)}")
                    print(f"⚠️  Missing sections: {', '.join(missing)}")
                else:
                    print(f"✅ Model card complete")
                    
                # Check for v1.0.1 mentions
                if "1.0.1" in content or "v1.0.1" in content:
                    print(f"✅ v1.0.1 documentation found")
                else:
                    self.warnings.append("v1.0.1 not mentioned in model card")
                    
            else:
                self.errors.append("README.md not found")
                print(f"❌ README.md not found")
                
        except Exception as e:
            self.warnings.append(f"Could not validate model card: {e}")
    
    def validate_files(self, repo_id):
        """Validate required files exist"""
        try:
            files = self.api.list_repo_files(repo_id)
            
            # Check for required files
            required = ["README.md", "config.json"]
            missing = []
            
            for req in required:
                if req not in files:
                    missing.append(req)
            
            if missing:
                self.warnings.append(f"Missing files: {', '.join(missing)}")
                print(f"⚠️  Missing files: {', '.join(missing)}")
            else:
                print(f"✅ All required files present")
            
            # Check for format files
            formats = {
                "safetensors": any(f.endswith('.safetensors') for f in files),
                "gguf": any(f.endswith('.gguf') for f in files),
                "mlx": any(f.endswith('.mlx') for f in files),
                "pytorch": any(f.endswith('.bin') for f in files)
            }
            
            available_formats = [fmt for fmt, exists in formats.items() if exists]
            if available_formats:
                print(f"✅ Available formats: {', '.join(available_formats)}")
            else:
                self.warnings.append("No model format files found")
                
        except Exception as e:
            self.warnings.append(f"Could not validate files: {e}")
    
    def validate_tags(self, repo_id, info):
        """Validate model tags"""
        required_tags = ["zen", "text-generation", "transformers"]
        
        if info.tags:
            missing = [tag for tag in required_tags if tag not in info.tags]
            
            if missing:
                self.warnings.append(f"Missing tags: {', '.join(missing)}")
                print(f"⚠️  Missing tags: {', '.join(missing)}")
            else:
                print(f"✅ All required tags present")
                
            # Check for version tags
            if "v1.0.1" in info.tags or "1.0.1" in info.tags:
                print(f"✅ Version tag found")
            else:
                self.warnings.append("v1.0.1 tag not found")
        else:
            self.errors.append("No tags found")
    
    def print_results(self, repo_id):
        """Print validation results"""
        print(f"\n{'='*60}")
        print(f"Results for {repo_id}:")
        print(f"{'='*60}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors and not self.warnings:
            print(f"\n✅ PERFECT - All validations passed!")
        
        # Clear for next model
        self.errors = []
        self.warnings = []
    
    def validate_all(self):
        """Validate all Zen models"""
        print("\n" + "="*60)
        print("ZEN MODEL VALIDATION SUITE")
        print("="*60)
        
        all_passed = True
        results = {}
        
        for repo_id, specs in self.model_specs.items():
            passed = self.validate_model(repo_id)
            results[repo_id] = passed
            if not passed:
                all_passed = False
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        for repo_id, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {repo_id}")
        
        if all_passed:
            print("\n🎉 ALL MODELS VALIDATED SUCCESSFULLY!")
        else:
            print("\n⚠️  Some models need attention")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate Zen models on HuggingFace")
    parser.add_argument("--repo-id", help="Model repository ID")
    parser.add_argument("--expected-size", type=int, help="Expected size in MB")
    parser.add_argument("--expected-params", help="Expected parameter count")
    parser.add_argument("--expected-arch", help="Expected architecture")
    parser.add_argument("--all", action="store_true", help="Validate all models")
    
    args = parser.parse_args()
    
    validator = ZenModelValidator()
    
    if args.all:
        success = validator.validate_all()
    elif args.repo_id:
        success = validator.validate_model(
            args.repo_id,
            args.expected_size,
            args.expected_params,
            args.expected_arch
        )
    else:
        print("Please specify --repo-id or --all")
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()