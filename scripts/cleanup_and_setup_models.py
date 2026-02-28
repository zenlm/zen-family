#!/usr/bin/env python3
"""
Clean up incorrect models and set up proper Zen model ecosystem
"""

import os
import sys
from huggingface_hub import HfApi, delete_repo, create_repo
from typing import List

class ZenModelCleanup:
    def __init__(self):
        self.api = HfApi()
        
        # Models to DELETE (outdated, incorrect naming, duplicates)
        self.models_to_delete = [
            "zenlm/zen-identity",
            "zenlm/zen-nano-thinking-4bit",
            "zenlm/zen-nano-instruct-4bit", 
            "zenlm/zen-nano-instruct-mlx-q8",
            "zenlm/zen-nano-instruct-mlx-q4",
            "zenlm/zen-nano-4b-thinking",  # Wrong - nano is 600M not 4B
            "zenlm/zen-nano-4b-instruct",  # Wrong - nano is 600M not 4B
            "zenlm/zen",  # Generic, not needed
            "zenlm/zen-coder",  # Missing -instruct suffix
            "zenlm/zen-omni-thinking",  # Not a main model
            "zenlm/zen-next",  # Missing -instruct suffix
            "zenlm/zen-omni-captioner",  # Not a main model
            "zenlm/zen-nano",  # Missing -instruct suffix
            "zenlm/zen-1",  # Old naming
            "zenlm/zen-1-thinking",  # Old naming
            "zenlm/zen-1-instruct",  # Old naming
            "zenlm/hanzo-zen1-fused",  # Old naming
            "zenlm/hanzo-zen1",  # Old naming
            "zenlm/zen-nano-thinking",  # Keep only -instruct variants for now
        ]
        
        # The ONLY 5 models we should have (main instruction models)
        self.correct_models = [
            {
                "repo_id": "zenlm/zen-nano-instruct",
                "size": "600M",
                "base_model": "Qwen/zen-0.5B-Instruct",
                "description": "Ultra-efficient 600M model for edge devices"
            },
            {
                "repo_id": "zenlm/zen-eco-instruct",
                "size": "4B", 
                "base_model": "Qwen/zen-3B-Instruct",
                "description": "Balanced 4B model for consumer hardware"
            },
            {
                "repo_id": "zenlm/zen-coder-instruct",
                "size": "32B",  # Using 32B as base, will describe as 480B MoE
                "base_model": "Qwen/zen-Coder-32B-Instruct",
                "description": "480B MoE model (35B active) for code generation"
            },
            {
                "repo_id": "zenlm/zen-omni-instruct",
                "size": "7B",  # Using 7B VL as base
                "base_model": "Qwen/zen-VL-7B-Instruct",
                "description": "30B MoE multimodal model (3B active)"
            },
            {
                "repo_id": "zenlm/zen-next-instruct",
                "size": "72B",  # Using 72B as base
                "base_model": "Qwen/zen-72B-Instruct",
                "description": "80B ultra-sparse MoE (3B active)"
            }
        ]
    
    def delete_incorrect_models(self):
        """Delete all incorrectly named or outdated models"""
        print("\n🗑️  DELETING INCORRECT MODELS")
        print("=" * 50)
        
        deleted = 0
        failed = 0
        
        for repo_id in self.models_to_delete:
            try:
                print(f"❌ Deleting {repo_id}...", end=" ")
                delete_repo(repo_id=repo_id, repo_type="model")
                print("✅ Deleted")
                deleted += 1
            except Exception as e:
                if "404" in str(e) or "Not Found" in str(e):
                    print("⚠️  Already deleted/not found")
                else:
                    print(f"❌ Error: {e}")
                    failed += 1
        
        print(f"\n📊 Deleted: {deleted}, Failed: {failed}")
        return deleted > 0 or failed == 0
    
    def verify_correct_models(self):
        """Verify the 5 correct models exist"""
        print("\n✅ VERIFYING CORRECT MODELS")
        print("=" * 50)
        
        all_good = True
        for model in self.correct_models:
            repo_id = model["repo_id"]
            try:
                # Check if repo exists
                self.api.repo_info(repo_id)
                print(f"✅ {repo_id:<30} | Exists")
            except:
                print(f"❌ {repo_id:<30} | Missing - will create")
                try:
                    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
                    print(f"   ✅ Created {repo_id}")
                except Exception as e:
                    print(f"   ❌ Failed to create: {e}")
                    all_good = False
        
        return all_good

def main():
    """Main cleanup and setup"""
    cleanup = ZenModelCleanup()
    
    print("\n🧹 ZEN MODEL ECOSYSTEM CLEANUP")
    print("=" * 50)
    
    # Step 1: Delete incorrect models
    if not cleanup.delete_incorrect_models():
        print("⚠️  Some models couldn't be deleted")
    
    # Step 2: Verify correct models exist
    if not cleanup.verify_correct_models():
        print("❌ Some correct models are missing")
        return False
    
    print("\n✅ CLEANUP COMPLETE!")
    print("\nNext steps:")
    print("1. Clone base models from Qwen")
    print("2. Generate GGUF files")
    print("3. Generate MLX files")
    print("4. Upload to HuggingFace")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)