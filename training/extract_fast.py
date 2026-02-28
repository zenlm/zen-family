#!/usr/bin/env python3
"""
Fast Full Git History Extraction
Uses find command for fast repo discovery
"""

import json
import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

OUTPUT_DIR = Path(__file__).parent / 'data' / 'full_history'
MAX_DIFF_SIZE = 50000  # 50KB per diff

def sanitize(text):
    if not isinstance(text, str):
        return str(text) if text else ""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]', text)
    text = re.sub(r'AKIA[0-9A-Z]{16}', '[AWS_KEY]', text)
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

def extract_repo(repo_path):
    """Extract from single repo"""
    repo_path = Path(repo_path)
    repo_name = f"{repo_path.parent.name}_{repo_path.name}"
    output_file = OUTPUT_DIR / f"{repo_name}.jsonl"

    # Skip if exists
    if output_file.exists() and output_file.stat().st_size > 100:
        return repo_name, 0, 0, "exists"

    examples = []
    total_bytes = 0

    try:
        # Get commits
        result = subprocess.run(
            ['git', 'log', '--pretty=format:%H|||%s|||%ai', '-n5000'],
            cwd=repo_path, capture_output=True, text=True, timeout=30, errors='replace'
        )
        if result.returncode != 0:
            return repo_name, 0, 0, "no commits"

        commits = [l for l in result.stdout.strip().split('\n') if l and '|||' in l]

        for line in commits:
            try:
                parts = line.split('|||')
                if len(parts) < 3:
                    continue
                sha, message, date = parts[0], parts[1], parts[2]

                # Get diff
                diff_result = subprocess.run(
                    ['git', 'show', '--patch', '--no-color', sha],
                    cwd=repo_path, capture_output=True, text=True, timeout=20, errors='replace'
                )
                if diff_result.returncode != 0:
                    continue

                diff = diff_result.stdout
                if len(diff) > MAX_DIFF_SIZE:
                    diff = diff[:MAX_DIFF_SIZE] + "\n...[truncated]"

                diff = sanitize(diff)
                message = sanitize(message)
                stage = get_stage(date)

                total_bytes += len(diff)

                example = {
                    "messages": [
                        {"role": "system", "content": f"You are Z, a {stage.replace('_', ' ')} developer."},
                        {"role": "user", "content": f"Implement: {message}"},
                        {"role": "assistant", "content": f"```diff\n{diff}\n```"}
                    ],
                    "metadata": {
                        "sha": sha[:8], "repo": repo_name, "date": date,
                        "stage": stage, "type": "git_full_diff"
                    }
                }
                examples.append(example)
            except:
                continue

        if examples:
            with open(output_file, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')

        return repo_name, len(examples), total_bytes, None

    except Exception as e:
        return repo_name, 0, 0, str(e)[:50]

def find_repos():
    """Use find command for fast discovery"""
    result = subprocess.run(
        ['find', os.path.expanduser('~/work'), os.path.expanduser('~/play'),
         '-maxdepth', '4', '-name', '.git', '-type', 'd'],
        capture_output=True, text=True, timeout=60
    )

    repos = []
    for line in result.stdout.strip().split('\n'):
        if line and 'node_modules' not in line and 'vendor' not in line and '.venv' not in line:
            repos.append(Path(line).parent)

    return sorted(set(repos))

def main():
    print("="*70)
    print("ZEN-AGENTIC: FAST FULL GIT EXTRACTION")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Finding repos (using find command)...")
    repos = find_repos()
    print(f"   Found {len(repos):,} repositories")

    print(f"\n🚀 Extracting (parallel, {os.cpu_count()} workers)...")

    total_commits = 0
    total_bytes = 0
    processed = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(extract_repo, r): r for r in repos}

        for future in as_completed(futures):
            name, commits, bytes_count, error = future.result()
            processed += 1
            total_commits += commits
            total_bytes += bytes_count

            if processed % 50 == 0 or processed == len(repos):
                gb = total_bytes / 1024**3
                tokens = total_bytes // 4
                print(f"   [{processed:,}/{len(repos):,}] Commits: {total_commits:,} | "
                      f"Tokens: {tokens/1e9:.2f}B | Size: {gb:.2f}GB")

    # Summary
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print(f"   Repos: {processed:,}")
    print(f"   Commits: {total_commits:,}")
    print(f"   Tokens: ~{total_bytes//4:,} ({total_bytes//4/1e9:.2f}B)")
    print(f"   Size: {total_bytes/1024**3:.2f} GB")
    print(f"   Output: {OUTPUT_DIR}")

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump({
            "repos": processed, "commits": total_commits,
            "bytes": total_bytes, "tokens": total_bytes // 4,
            "completed": datetime.now().isoformat()
        }, f, indent=2)

if __name__ == '__main__':
    main()
