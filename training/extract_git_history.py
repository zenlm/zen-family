#!/usr/bin/env python3
"""
Extract Git History Training Data

Converts git commits into training examples for fine-tuning LLMs.
Each commit becomes a training example showing code changes.

Training format:
- Input: Commit message + context (file list, repo info)
- Output: The actual diff/changes

This creates a "pure Z AI dev" model based on your heroic coding career.
"""

import json
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import re

def sanitize(text):
    """Remove PII and sensitive data"""
    if not isinstance(text, str):
        return str(text) if text else ""
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Remove API keys
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]', text)
    text = re.sub(r'gho_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]', text)
    # Remove IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    # Remove common secrets patterns
    text = re.sub(r'(password|secret|token|key)\s*[=:]\s*["\'][^"\']+["\']', r'\1=[REDACTED]', text, flags=re.IGNORECASE)
    return text

def get_career_stage(date_str):
    """Determine career stage based on commit date"""
    if not date_str:
        return "unknown", 0

    try:
        # Parse various date formats
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                date = datetime.strptime(date_str[:19], fmt[:len(date_str[:19])])
                break
            except:
                continue
        else:
            date = datetime.strptime(date_str[:10], "%Y-%m-%d")

        year = date.year

        # Career stages based on evolution
        if year <= 2010:
            return "early_career", 1  # Foundational years
        elif year <= 2013:
            return "growth", 2  # Building expertise
        elif year <= 2016:
            return "senior", 3  # Senior developer
        elif year <= 2019:
            return "architect", 4  # Architecture & leadership
        elif year <= 2022:
            return "principal", 5  # Principal/Staff level
        else:
            return "frontier", 6  # Cutting-edge AI/blockchain

    except Exception:
        return "unknown", 0

def extract_from_stats_db(db_path, output_path):
    """Extract training data from the stats.db SQLite database"""
    print(f"📊 Loading commits from {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"   Tables: {tables}")

    # Get commits ordered by date (oldest first for evolution tracking)
    cursor.execute("""
        SELECT sha, message, username, repo, additions, deletions, date
        FROM commits
        ORDER BY date ASC
    """)

    commits = cursor.fetchall()
    print(f"   Found {len(commits):,} commits")

    # Create training examples
    examples = []

    # Track stats by career stage
    stage_stats = {}

    for idx, commit in enumerate(tqdm(commits, desc="Processing commits")):
        sha = commit['sha']
        message = sanitize(commit['message'] or "")
        repo = commit['repo'] or ""
        additions = commit['additions'] or 0
        deletions = commit['deletions'] or 0
        date = commit['date'] or ""

        # Skip commits with no message
        if not message or len(message) < 5:
            continue

        # Get career stage for evolution tracking
        career_stage, stage_num = get_career_stage(date)

        # Track stage stats
        if career_stage not in stage_stats:
            stage_stats[career_stage] = {"count": 0, "additions": 0, "deletions": 0}
        stage_stats[career_stage]["count"] += 1
        stage_stats[career_stage]["additions"] += additions
        stage_stats[career_stage]["deletions"] += deletions

        # Determine project category from repo name
        repo_lower = repo.lower() if repo else ""
        if any(x in repo_lower for x in ['ai', 'ml', 'model', 'llm', 'neural', 'zen']):
            domain = "ai_ml"
        elif any(x in repo_lower for x in ['lux', 'chain', 'crypto', 'wallet', 'defi', 'nft']):
            domain = "blockchain"
        elif any(x in repo_lower for x in ['web', 'react', 'vue', 'next', 'ui', 'frontend']):
            domain = "frontend"
        elif any(x in repo_lower for x in ['api', 'server', 'backend', 'node']):
            domain = "backend"
        else:
            domain = "general"

        # Create training example with evolution context
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are Z, a {career_stage.replace('_', ' ')} software developer. Your expertise spans blockchain, AI, and full-stack development. You write clean, efficient code."
                },
                {
                    "role": "user",
                    "content": f"Implement the following change to {repo}:\n\n{message}"
                },
                {
                    "role": "assistant",
                    "content": f"I'll implement this change.\n\n**Changes Made**:\n- Modified {repo} with {additions:,} lines added and {deletions:,} lines removed\n- {message}\n\nThe implementation follows best practices for {domain} development."
                }
            ],
            "metadata": {
                "sha": sha[:8],
                "repo": repo,
                "additions": additions,
                "deletions": deletions,
                "date": date,
                "career_stage": career_stage,
                "stage_num": stage_num,
                "domain": domain,
                "commit_index": idx,
                "type": "git_commit"
            }
        }
        examples.append(example)

    conn.close()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    total_additions = sum(e['metadata']['additions'] for e in examples)
    total_deletions = sum(e['metadata']['deletions'] for e in examples)

    print(f"\n✅ Extracted {len(examples):,} training examples")
    print(f"   Total lines: +{total_additions:,} / -{total_deletions:,}")
    print(f"   Output: {output_path}")

    # Print career stage breakdown
    print(f"\n📈 Career Evolution Breakdown:")
    stage_order = ["early_career", "growth", "senior", "architect", "principal", "frontier", "unknown"]
    for stage in stage_order:
        if stage in stage_stats:
            s = stage_stats[stage]
            print(f"   {stage:15} | {s['count']:>6,} commits | +{s['additions']:>10,} / -{s['deletions']:>10,} lines")

    return examples

def extract_from_local_repos(work_dir, output_path, author_email=None):
    """Extract training data from local git repos with full diffs"""
    work_dir = Path(work_dir)
    output_path = Path(output_path)

    # Find all git repos
    repos = []
    for git_dir in work_dir.rglob('.git'):
        if git_dir.is_dir():
            repos.append(git_dir.parent)

    print(f"📁 Found {len(repos)} local repositories")

    examples = []

    for repo_path in tqdm(repos, desc="Processing repos"):
        try:
            # Get commits with diffs
            cmd = ['git', 'log', '--pretty=format:%H|||%s|||%an|||%ae|||%ai', '-n', '1000']
            if author_email:
                cmd.extend(['--author', author_email])

            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                continue

            for line in result.stdout.strip().split('\n'):
                if not line or '|||' not in line:
                    continue

                parts = line.split('|||')
                if len(parts) < 5:
                    continue

                sha, message, author, email, date = parts[:5]

                # Get diff for this commit
                diff_result = subprocess.run(
                    ['git', 'show', '--stat', sha],
                    cwd=repo_path, capture_output=True, text=True, timeout=30
                )

                diff_stat = diff_result.stdout if diff_result.returncode == 0 else ""

                # Extract files changed
                files_changed = []
                additions = 0
                deletions = 0

                for diff_line in diff_stat.split('\n'):
                    if '|' in diff_line and ('+' in diff_line or '-' in diff_line):
                        files_changed.append(diff_line.split('|')[0].strip())
                    if 'insertion' in diff_line or 'deletion' in diff_line:
                        # Parse "X files changed, Y insertions(+), Z deletions(-)"
                        nums = re.findall(r'(\d+)\s+insertion', diff_line)
                        if nums:
                            additions = int(nums[0])
                        nums = re.findall(r'(\d+)\s+deletion', diff_line)
                        if nums:
                            deletions = int(nums[0])

                repo_name = repo_path.name

                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are Z, an expert software developer with decades of experience across blockchain, AI, and full-stack development."
                        },
                        {
                            "role": "user",
                            "content": f"Make changes to {repo_name}: {sanitize(message)}"
                        },
                        {
                            "role": "assistant",
                            "content": f"I'll implement this change.\n\n**Files modified**:\n" +
                                      "\n".join(f"- {f}" for f in files_changed[:10]) +
                                      f"\n\n**Changes**: +{additions} lines, -{deletions} lines\n\n" +
                                      f"The implementation follows the codebase patterns and best practices."
                        }
                    ],
                    "metadata": {
                        "sha": sha[:8],
                        "repo": repo_name,
                        "additions": additions,
                        "deletions": deletions,
                        "date": date,
                        "files": files_changed[:10],
                        "type": "git_commit_local"
                    }
                }
                examples.append(example)

        except Exception as e:
            continue

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    total_additions = sum(e['metadata']['additions'] for e in examples)
    total_deletions = sum(e['metadata']['deletions'] for e in examples)

    print(f"\n✅ Extracted {len(examples):,} training examples from local repos")
    print(f"   Total lines: +{total_additions:,} / -{total_deletions:,}")
    print(f"   Output: {output_path}")

    return examples

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║     ZEN-AGENTIC: GIT HISTORY EXTRACTION                          ║
║     Creating training data from your heroic coding career        ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    data_dir = Path(__file__).parent / 'data'

    # Extract from stats.db (GitHub API data with full history)
    stats_db = Path.home() / 'work' / 'zeekay' / 'stats' / 'cache' / 'stats.db'
    if stats_db.exists():
        print("\n📊 Extracting from GitHub stats database...")
        extract_from_stats_db(stats_db, data_dir / 'git_history_github.jsonl')

    # Extract from local repos (with more detail)
    print("\n📁 Extracting from local repositories...")
    extract_from_local_repos(
        Path.home() / 'work',
        data_dir / 'git_history_local.jsonl'
    )

    print("\n" + "="*70)
    print("✅ Git history extraction complete!")
    print("   Run upload script to push to HuggingFace")

if __name__ == '__main__':
    main()
