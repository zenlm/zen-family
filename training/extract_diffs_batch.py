#!/usr/bin/env python3
"""
Fast batch diff extraction from key repos
Processes specific high-value directories
"""

import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / 'data' / 'diffs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sanitize(text):
    """Remove PII"""
    if not isinstance(text, str):
        return str(text) if text else ""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    text = re.sub(r'-----BEGIN[^-]+PRIVATE KEY-----[\s\S]*?-----END[^-]+PRIVATE KEY-----', '[PRIVATE_KEY]', text)
    return text

def get_stage(date_str):
    try:
        year = int(date_str[:4])
        if year <= 2010: return "early_career"
        elif year <= 2013: return "growth"
        elif year <= 2016: return "senior"
        elif year <= 2019: return "architect"
        elif year <= 2022: return "principal"
        else: return "frontier"
    except:
        return "unknown"

def extract_repo(repo_path, max_commits=500):
    """Extract diffs from a repo"""
    repo_path = Path(repo_path)
    if not (repo_path / '.git').exists():
        return [], 0

    repo_name = repo_path.name
    examples = []
    total_chars = 0

    try:
        # Get commits
        result = subprocess.run(
            ['git', 'log', '--pretty=format:%H|||%s|||%ai', f'-n{max_commits}'],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return [], 0

        for line in result.stdout.strip().split('\n'):
            if not line or '|||' not in line:
                continue

            parts = line.split('|||')
            if len(parts) < 3:
                continue

            sha, message, date = parts[0], parts[1], parts[2]

            # Get diff (limit size)
            diff_result = subprocess.run(
                ['git', 'show', '--stat', '--patch', '--no-color', sha],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )

            if diff_result.returncode != 0:
                continue

            diff = diff_result.stdout
            if len(diff) > 30000:
                diff = diff[:30000] + "\n...[truncated]"

            diff = sanitize(diff)
            message = sanitize(message)
            stage = get_stage(date)

            total_chars += len(diff)

            examples.append({
                "messages": [
                    {"role": "system", "content": f"You are Z, a {stage.replace('_', ' ')} developer."},
                    {"role": "user", "content": f"Implement: {message}"},
                    {"role": "assistant", "content": f"```diff\n{diff}\n```"}
                ],
                "metadata": {
                    "sha": sha[:8],
                    "repo": repo_name,
                    "date": date,
                    "stage": stage,
                    "type": "git_diff"
                }
            })

    except Exception as e:
        print(f"   Error: {e}")

    return examples, total_chars

def main():
    # Key directories to process
    key_dirs = [
        Path.home() / 'work' / 'lux',
        Path.home() / 'work' / 'hanzo',
        Path.home() / 'work' / 'zen',
        Path.home() / 'work' / 'zoo',
        Path.home() / 'work' / 'zeekay',
    ]

    total_examples = 0
    total_tokens = 0
    all_examples = []

    for base_dir in key_dirs:
        if not base_dir.exists():
            continue

        print(f"\n📁 Processing {base_dir.name}/")

        # Find repos in this directory
        repos = []
        for item in base_dir.iterdir():
            if item.is_dir() and (item / '.git').exists():
                repos.append(item)

        print(f"   Found {len(repos)} repos")

        for repo in sorted(repos):
            examples, chars = extract_repo(repo)
            if examples:
                all_examples.extend(examples)
                tokens = chars // 4
                total_examples += len(examples)
                total_tokens += tokens
                print(f"   ✓ {repo.name}: {len(examples)} commits, ~{tokens:,} tokens")

    # Save combined output
    output_file = OUTPUT_DIR / 'key_repos_diffs.jsonl'
    with open(output_file, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\n{'='*60}")
    print(f"✅ COMPLETE")
    print(f"   Total examples: {total_examples:,}")
    print(f"   Total tokens: ~{total_tokens:,}")
    print(f"   Output: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    main()
