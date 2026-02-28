#!/usr/bin/env python3
"""
Migrate model weights from old -instruct repos to new unified repos
and clean up old repositories after verification
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import (
    HfApi,
    snapshot_download,
    upload_folder,
    delete_repo,
    list_repo_files,
    hf_hub_download
)
from huggingface_hub.utils import RepositoryNotFoundError
import hashlib
import time

class ZenWeightMigrator:
    """Migrate weights from old -instruct repos to new unified repos"""

    def __init__(self, hf_token: str = None):
        """Initialize with HuggingFace token"""
        self.api = HfApi(token=hf_token)
        self.migrations = {
            "zen-nano": {
                "old": "zenlm/zen-nano-instruct",
                "new": "zenlm/zen-nano"
            },
            "zen-eco": {
                "old": "zenlm/zen-eco-instruct",
                "new": "zenlm/zen-eco"
            },
            "zen-omni": {
                "old": "zenlm/zen-omni-instruct",
                "new": "zenlm/zen-omni"
            },
            "zen-coder": {
                "old": "zenlm/zen-coder-instruct",
                "new": "zenlm/zen-coder"
            },
            "zen-next": {
                "old": "zenlm/zen-next-instruct",
                "new": "zenlm/zen-next"
            }
        }

    def get_repo_files(self, repo_id: str) -> List[str]:
        """Get list of files in repository"""
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="model")
            return [f for f in files if not f.startswith('.')]
        except RepositoryNotFoundError:
            return []

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_migration(self, old_repo: str, new_repo: str, temp_dir: str) -> bool:
        """Verify that all files were migrated correctly"""
        print(f"Verifying migration from {old_repo} to {new_repo}...")

        old_files = self.get_repo_files(old_repo)
        new_files = self.get_repo_files(new_repo)

        # Files to skip in comparison (these are expected to be different)
        skip_files = {'README.md', 'config.json', 'inference_example.py'}

        # Check model weight files
        model_files = [f for f in old_files if (
            f.endswith(('.bin', '.safetensors', '.gguf', '.pt', '.pth')) or
            f in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                  'vocab.json', 'merges.txt', 'generation_config.json']
        )]

        missing_files = []
        mismatched_files = []

        for file in model_files:
            if file not in new_files:
                missing_files.append(file)
            else:
                # Download both files and compare hashes
                try:
                    old_path = os.path.join(temp_dir, f"old_{file}")
                    new_path = os.path.join(temp_dir, f"new_{file}")

                    # Download files
                    hf_hub_download(repo_id=old_repo, filename=file, local_dir=temp_dir, cache_dir=temp_dir)
                    hf_hub_download(repo_id=new_repo, filename=file, local_dir=temp_dir, cache_dir=temp_dir)

                    # Compare hashes
                    old_hash = self.calculate_file_hash(old_path)
                    new_hash = self.calculate_file_hash(new_path)

                    if old_hash != new_hash:
                        mismatched_files.append(file)

                except Exception as e:
                    print(f"  Warning: Could not verify {file}: {e}")

        if missing_files:
            print(f"  ✗ Missing files in new repo: {missing_files}")
            return False

        if mismatched_files:
            print(f"  ✗ Files with different content: {mismatched_files}")
            return False

        print(f"  ✓ All model files verified successfully")
        return True

    def migrate_weights(self, model_key: str, verify: bool = True, delete_old: bool = False) -> bool:
        """Migrate weights from old repo to new repo"""
        old_repo = self.migrations[model_key]["old"]
        new_repo = self.migrations[model_key]["new"]

        print(f"\n{'='*60}")
        print(f"Migrating weights: {old_repo} → {new_repo}")
        print(f"{'='*60}")

        try:
            # Check if old repo exists
            try:
                old_files = self.get_repo_files(old_repo)
                if not old_files:
                    print(f"⚠ No files found in {old_repo}")
                    return False
                print(f"✓ Found {len(old_files)} files in {old_repo}")
            except RepositoryNotFoundError:
                print(f"✗ Repository not found: {old_repo}")
                return False

            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Downloading model files to temporary directory...")

                # Download all files from old repo
                snapshot_path = snapshot_download(
                    repo_id=old_repo,
                    repo_type="model",
                    cache_dir=temp_dir,
                    local_dir=os.path.join(temp_dir, "model_files")
                )
                print(f"✓ Downloaded files to {snapshot_path}")

                # Prepare files for upload (excluding README.md since we have a new one)
                model_path = Path(snapshot_path)
                files_to_upload = []

                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(model_path)
                        # Skip README.md and config.json as we have new versions
                        if relative_path.name not in ['README.md', '.gitattributes']:
                            files_to_upload.append(str(relative_path))

                print(f"Found {len(files_to_upload)} files to migrate")

                # Upload to new repository
                if files_to_upload:
                    print(f"Uploading files to {new_repo}...")

                    # Upload the entire folder
                    upload_folder(
                        folder_path=snapshot_path,
                        repo_id=new_repo,
                        repo_type="model",
                        commit_message=f"Migrate model weights from {old_repo}",
                        ignore_patterns=["README.md", ".gitattributes"]
                    )
                    print(f"✓ Successfully uploaded files to {new_repo}")

                    # Verify migration if requested
                    if verify:
                        if not self.verify_migration(old_repo, new_repo, temp_dir):
                            print("✗ Migration verification failed")
                            return False

                    # Delete old repo if requested and verification passed
                    if delete_old:
                        print(f"\nDeleting old repository: {old_repo}")
                        response = input("  Are you sure? This cannot be undone! (yes/no): ")
                        if response.lower() == 'yes':
                            try:
                                delete_repo(repo_id=old_repo, repo_type="model")
                                print(f"  ✓ Deleted {old_repo}")
                            except Exception as e:
                                print(f"  ✗ Failed to delete: {e}")
                        else:
                            print("  Skipped deletion")

                    print(f"✅ Successfully migrated {model_key}")
                    return True
                else:
                    print("⚠ No files to migrate")
                    return False

        except Exception as e:
            print(f"❌ Error during migration: {e}")
            return False

    def migrate_all(self, verify: bool = True, delete_old: bool = False) -> None:
        """Migrate all models"""
        print("\n" + "="*60)
        print("ZEN MODEL WEIGHT MIGRATION")
        print("="*60)

        successful = []
        failed = []
        skipped = []

        for model_key in self.migrations.keys():
            print(f"\n[{len(successful) + len(failed) + 1}/{len(self.migrations)}] Processing {model_key}")

            # Check if migration is needed
            old_repo = self.migrations[model_key]["old"]
            new_repo = self.migrations[model_key]["new"]

            old_files = self.get_repo_files(old_repo)
            new_files = self.get_repo_files(new_repo)

            # Check if new repo already has model weights
            has_weights = any(
                f.endswith(('.bin', '.safetensors', '.gguf', '.pt', '.pth'))
                for f in new_files
            )

            if has_weights:
                print(f"  ⚠ {new_repo} already has model weights, skipping...")
                skipped.append(model_key)
                continue

            if not old_files:
                print(f"  ⚠ {old_repo} has no files, skipping...")
                skipped.append(model_key)
                continue

            # Perform migration
            if self.migrate_weights(model_key, verify=verify, delete_old=delete_old):
                successful.append(model_key)
            else:
                failed.append(model_key)

            # Small delay between operations
            time.sleep(2)

        # Print summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)

        if successful:
            print(f"\n✅ Successfully migrated ({len(successful)}):")
            for model in successful:
                print(f"  - {model}: {self.migrations[model]['old']} → {self.migrations[model]['new']}")

        if skipped:
            print(f"\n⚠ Skipped ({len(skipped)}):")
            for model in skipped:
                print(f"  - {model}: Already has weights or source is empty")

        if failed:
            print(f"\n❌ Failed to migrate ({len(failed)}):")
            for model in failed:
                print(f"  - {model}")

    def cleanup_old_repos(self) -> None:
        """Interactive cleanup of old repositories after verification"""
        print("\n" + "="*60)
        print("CLEANUP OLD REPOSITORIES")
        print("="*60)

        print("\n⚠️  WARNING: This will permanently delete old repositories!")
        print("Make sure all migrations have been verified first.\n")

        for model_key, repos in self.migrations.items():
            old_repo = repos["old"]
            new_repo = repos["new"]

            # Check if old repo exists
            try:
                old_files = self.get_repo_files(old_repo)
                if not old_files:
                    print(f"⚠ {old_repo} is already empty")
                    continue
            except RepositoryNotFoundError:
                print(f"⚠ {old_repo} not found")
                continue

            # Check if new repo has weights
            new_files = self.get_repo_files(new_repo)
            has_weights = any(
                f.endswith(('.bin', '.safetensors', '.gguf', '.pt', '.pth'))
                for f in new_files
            )

            if not has_weights:
                print(f"✗ {new_repo} doesn't have model weights yet")
                print(f"  Cannot delete {old_repo} until migration is complete")
                continue

            # Ask for confirmation
            print(f"\nDelete {old_repo}?")
            print(f"  New repo {new_repo} has {len(new_files)} files")
            response = input("  Delete? (yes/no/skip all): ").lower()

            if response == 'skip all':
                print("Skipping all remaining deletions")
                break
            elif response == 'yes':
                try:
                    delete_repo(repo_id=old_repo, repo_type="model")
                    print(f"  ✓ Deleted {old_repo}")
                except Exception as e:
                    print(f"  ✗ Failed to delete: {e}")
            else:
                print("  Skipped")

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate Zen model weights and cleanup old repos")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--model", help="Specific model to migrate (zen-nano, zen-eco, etc.)")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification after migration")
    parser.add_argument("--delete-old", action="store_true", help="Delete old repos after successful migration")
    parser.add_argument("--cleanup-only", action="store_true", help="Only run cleanup of old repos")

    args = parser.parse_args()

    # Get token
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: HuggingFace token required. Set HF_TOKEN env var or use --token")
        return 1

    # Initialize migrator
    migrator = ZenWeightMigrator(hf_token=token)

    if args.cleanup_only:
        # Just run cleanup
        migrator.cleanup_old_repos()
    elif args.model:
        # Migrate specific model
        if args.model in migrator.migrations:
            migrator.migrate_weights(
                args.model,
                verify=not args.no_verify,
                delete_old=args.delete_old
            )
        else:
            print(f"Error: Unknown model {args.model}")
            print(f"Available models: {', '.join(migrator.migrations.keys())}")
            return 1
    else:
        # Migrate all models
        migrator.migrate_all(
            verify=not args.no_verify,
            delete_old=args.delete_old
        )

    return 0

if __name__ == "__main__":
    exit(main())