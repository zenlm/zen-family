#!/usr/bin/env python3
"""
FULL Git History Extraction - All Repos, All Commits, Full Diffs

Target: 1000+ repos, 100B+ tokens
"""

import json
import subprocess
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
OUTPUT_DIR = Path(__file__).parent / 'data' / 'full_history'
PROGRESS_FILE = OUTPUT_DIR / '.progress.json'
MAX_DIFF_SIZE = 100000  # 100KB per diff max
MAX_COMMITS_PER_REPO = 10000  # Get everything

# Thread-safe counters
lock = threading.Lock()
stats = {
    "repos_processed": 0,
    "repos_total": 0,
    "commits_total": 0,
    "tokens_total": 0,
    "bytes_total": 0,
    "errors": 0
}

def sanitize(text):
    """Remove PII"""
    if not isinstance(text, str):
        return str(text) if text else ""
    # Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # API keys
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]', text)
    text = re.sub(r'gho_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]', text)
    # AWS keys
    text = re.sub(r'AKIA[0-9A-Z]{16}', '[AWS_KEY]', text)
    # Private keys
    text = re.sub(r'-----BEGIN[^-]+PRIVATE KEY-----[\s\S]*?-----END[^-]+PRIVATE KEY-----', '[PRIVATE_KEY]', text)
    # IPs
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    return text

def get_stage(date_str):
    """Career stage from date"""
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

def detect_language(files_changed):
    """Detect language from changed files"""
    ext_map = {
        '.go': 'Go', '.py': 'Python', '.ts': 'TypeScript', '.tsx': 'TypeScript',
        '.js': 'JavaScript', '.jsx': 'JavaScript', '.rs': 'Rust', '.sol': 'Solidity',
        '.java': 'Java', '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift',
        '.kt': 'Kotlin', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
        '.ex': 'Elixir', '.exs': 'Elixir', '.hs': 'Haskell', '.scala': 'Scala'
    }

    lang_counts = {}
    for f in files_changed:
        ext = Path(f).suffix.lower()
        if ext in ext_map:
            lang = ext_map[ext]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if lang_counts:
        return max(lang_counts, key=lang_counts.get)
    return "Unknown"

def extract_repo(repo_path):
    """Extract full history from a single repo"""
    repo_path = Path(repo_path)
    repo_name = repo_path.name
    parent_name = repo_path.parent.name
    full_name = f"{parent_name}/{repo_name}"

    examples = []
    repo_tokens = 0
    repo_bytes = 0

    try:
        # Get ALL commits
        result = subprocess.run(
            ['git', 'log', '--pretty=format:%H|||%s|||%ai|||%an', f'-n{MAX_COMMITS_PER_REPO}'],
            cwd=repo_path, capture_output=True, text=True, timeout=60, errors='replace'
        )

        if result.returncode != 0:
            return [], 0, 0, f"git log failed"

        commits = [l for l in result.stdout.strip().split('\n') if l and '|||' in l]

        for line in commits:
            try:
                parts = line.split('|||')
                if len(parts) < 4:
                    continue

                sha, message, date, author = parts[0], parts[1], parts[2], parts[3]

                # Get FULL diff
                diff_result = subprocess.run(
                    ['git', 'show', '--patch', '--no-color', '-U3', sha],
                    cwd=repo_path, capture_output=True, text=True, timeout=30, errors='replace'
                )

                if diff_result.returncode != 0:
                    continue

                diff = diff_result.stdout

                # Truncate if too large but keep substantial content
                if len(diff) > MAX_DIFF_SIZE:
                    diff = diff[:MAX_DIFF_SIZE] + "\n\n... [truncated - full diff was larger]"

                # Sanitize
                diff = sanitize(diff)
                message = sanitize(message)

                # Extract files changed from diff
                files_changed = re.findall(r'^diff --git a/(.+?) b/', diff, re.MULTILINE)
                language = detect_language(files_changed)
                stage = get_stage(date)

                # Count stats
                additions = len(re.findall(r'^\+[^+]', diff, re.MULTILINE))
                deletions = len(re.findall(r'^-[^-]', diff, re.MULTILINE))

                diff_bytes = len(diff.encode('utf-8'))
                diff_tokens = diff_bytes // 4  # rough estimate
                repo_bytes += diff_bytes
                repo_tokens += diff_tokens

                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are Z, a {stage.replace('_', ' ')} {language} developer."
                        },
                        {
                            "role": "user",
                            "content": f"Implement this change to {full_name}: {message}"
                        },
                        {
                            "role": "assistant",
                            "content": f"```diff\n{diff}\n```"
                        }
                    ],
                    "metadata": {
                        "sha": sha[:8],
                        "repo": full_name,
                        "date": date,
                        "stage": stage,
                        "language": language,
                        "files": files_changed[:20],
                        "additions": additions,
                        "deletions": deletions,
                        "tokens": diff_tokens,
                        "type": "git_full_diff"
                    }
                }
                examples.append(example)

            except Exception as e:
                continue

        return examples, repo_tokens, repo_bytes, None

    except subprocess.TimeoutExpired:
        return [], 0, 0, "timeout"
    except Exception as e:
        return [], 0, 0, str(e)

def process_repo(repo_path, output_dir):
    """Process a single repo and save results"""
    global stats

    repo_name = f"{repo_path.parent.name}_{repo_path.name}"
    output_file = output_dir / f"{repo_name}.jsonl"

    # Skip if already processed
    if output_file.exists() and output_file.stat().st_size > 0:
        # Count existing
        with open(output_file) as f:
            count = sum(1 for _ in f)
        with lock:
            stats["repos_processed"] += 1
        return repo_name, count, 0, "skipped"

    examples, tokens, bytes_count, error = extract_repo(repo_path)

    if examples:
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

    with lock:
        stats["repos_processed"] += 1
        stats["commits_total"] += len(examples)
        stats["tokens_total"] += tokens
        stats["bytes_total"] += bytes_count
        if error:
            stats["errors"] += 1

    return repo_name, len(examples), tokens, error

def find_all_repos(directories):
    """Find all git repos in given directories"""
    repos = []
    for base_dir in directories:
        base = Path(base_dir).expanduser()
        if not base.exists():
            continue

        for git_dir in base.rglob('.git'):
            if git_dir.is_dir():
                repo = git_dir.parent
                # Skip nested repos in node_modules, vendor, etc
                path_str = str(repo)
                if any(skip in path_str for skip in ['node_modules', 'vendor', '.venv', 'venv', '__pycache__']):
                    continue
                repos.append(repo)

    return sorted(set(repos), key=lambda x: str(x))

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║     ZEN-AGENTIC: FULL GIT HISTORY EXTRACTION                              ║
║     Target: 1000+ repos, 100B+ tokens                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all repos
    print("🔍 Finding all repositories...")
    repos = find_all_repos(['~/work', '~/play'])
    stats["repos_total"] = len(repos)
    print(f"   Found {len(repos):,} repositories")

    # Process in parallel
    print(f"\n🚀 Extracting full git histories (parallel)...")
    print(f"   Output: {OUTPUT_DIR}")
    print()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_repo, repo, OUTPUT_DIR): repo for repo in repos}

        for i, future in enumerate(as_completed(futures)):
            repo_name, commits, tokens, error = future.result()

            # Progress every 10 repos
            if (i + 1) % 10 == 0 or i == len(repos) - 1:
                pct = (stats["repos_processed"] / stats["repos_total"]) * 100
                gb = stats["bytes_total"] / 1024 / 1024 / 1024
                print(f"   [{stats['repos_processed']:,}/{stats['repos_total']:,}] ({pct:.1f}%) | "
                      f"Commits: {stats['commits_total']:,} | "
                      f"Tokens: {stats['tokens_total']/1e9:.2f}B | "
                      f"Size: {gb:.2f}GB")

    # Final stats
    print(f"\n{'='*75}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*75}")
    print(f"   Repositories: {stats['repos_processed']:,}")
    print(f"   Total commits: {stats['commits_total']:,}")
    print(f"   Total tokens: {stats['tokens_total']:,} ({stats['tokens_total']/1e9:.2f}B)")
    print(f"   Total size: {stats['bytes_total']/1024/1024/1024:.2f} GB")
    print(f"   Errors: {stats['errors']}")
    print(f"   Output: {OUTPUT_DIR}")

    # Save summary
    summary = {
        "completed": datetime.now().isoformat(),
        "repos": stats["repos_processed"],
        "commits": stats["commits_total"],
        "tokens": stats["tokens_total"],
        "bytes": stats["bytes_total"],
        "errors": stats["errors"]
    }
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
