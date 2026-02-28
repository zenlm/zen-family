#!/usr/bin/env python3
"""
Incremental Git History Extraction with Full Diffs

Features:
- Processes repos one by one, saves progress
- Includes actual code diffs (not just metadata)
- Rich metadata: language, framework, domain
- Idempotent - can resume from where it left off
- Uploads to HuggingFace periodically
"""

import json
import sqlite3
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Progress tracking
PROGRESS_FILE = Path(__file__).parent / 'data' / '.extraction_progress.json'
OUTPUT_DIR = Path(__file__).parent / 'data' / 'git_diffs'

def load_progress():
    """Load extraction progress"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "processed_repos": [],
        "total_commits": 0,
        "total_tokens": 0,
        "total_additions": 0,
        "total_deletions": 0,
        "by_language": {},
        "by_domain": {},
        "by_stage": {},
        "last_updated": None
    }

def save_progress(progress):
    """Save extraction progress"""
    progress["last_updated"] = datetime.now().isoformat()
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

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
    # Remove secrets
    text = re.sub(r'(password|secret|token|key)\s*[=:]\s*["\'][^"\']+["\']', r'\1=[REDACTED]', text, flags=re.IGNORECASE)
    # Remove private keys
    text = re.sub(r'-----BEGIN[^-]+PRIVATE KEY-----[\s\S]*?-----END[^-]+PRIVATE KEY-----', '[PRIVATE_KEY]', text)
    return text

def detect_language(repo_path):
    """Detect primary language from file extensions"""
    extensions = defaultdict(int)
    try:
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and vendor dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'vendor', 'venv', '.venv']]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext:
                    extensions[ext] += 1
    except:
        pass

    # Map extensions to languages
    lang_map = {
        '.go': 'Go', '.py': 'Python', '.ts': 'TypeScript', '.tsx': 'TypeScript',
        '.js': 'JavaScript', '.jsx': 'JavaScript', '.rs': 'Rust', '.sol': 'Solidity',
        '.java': 'Java', '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift',
        '.kt': 'Kotlin', '.cpp': 'C++', '.c': 'C', '.cs': 'C#', '.ex': 'Elixir',
        '.exs': 'Elixir', '.hs': 'Haskell', '.scala': 'Scala', '.clj': 'Clojure'
    }

    lang_counts = defaultdict(int)
    for ext, count in extensions.items():
        if ext in lang_map:
            lang_counts[lang_map[ext]] += count

    if lang_counts:
        return max(lang_counts, key=lang_counts.get)
    return "Unknown"

def detect_framework(repo_path):
    """Detect framework from config files"""
    frameworks = []

    checks = [
        ('package.json', ['next', 'react', 'vue', 'angular', 'express', 'fastify']),
        ('Cargo.toml', ['tokio', 'actix', 'rocket', 'axum']),
        ('go.mod', ['gin', 'echo', 'fiber', 'chi']),
        ('requirements.txt', ['django', 'flask', 'fastapi', 'pytorch', 'tensorflow']),
        ('pyproject.toml', ['django', 'flask', 'fastapi', 'pytorch', 'tensorflow']),
    ]

    for config_file, framework_list in checks:
        config_path = repo_path / config_file
        if config_path.exists():
            try:
                content = config_path.read_text().lower()
                for fw in framework_list:
                    if fw in content:
                        frameworks.append(fw.capitalize())
            except:
                pass

    return frameworks[:3] if frameworks else ["None"]

def detect_domain(repo_name):
    """Detect project domain from repo name"""
    repo_lower = repo_name.lower()

    if any(x in repo_lower for x in ['ai', 'ml', 'model', 'llm', 'neural', 'zen', 'gpt', 'bert']):
        return "ai_ml"
    elif any(x in repo_lower for x in ['lux', 'chain', 'crypto', 'wallet', 'defi', 'nft', 'eth', 'sol']):
        return "blockchain"
    elif any(x in repo_lower for x in ['web', 'react', 'vue', 'next', 'ui', 'frontend', 'app']):
        return "frontend"
    elif any(x in repo_lower for x in ['api', 'server', 'backend', 'node', 'service']):
        return "backend"
    elif any(x in repo_lower for x in ['infra', 'deploy', 'docker', 'k8s', 'terraform']):
        return "infrastructure"
    elif any(x in repo_lower for x in ['cli', 'tool', 'util']):
        return "tooling"
    else:
        return "general"

def get_career_stage(date_str):
    """Determine career stage based on commit date"""
    if not date_str:
        return "unknown", 0
    try:
        date = datetime.strptime(date_str[:10], "%Y-%m-%d")
        year = date.year
        if year <= 2010:
            return "early_career", 1
        elif year <= 2013:
            return "growth", 2
        elif year <= 2016:
            return "senior", 3
        elif year <= 2019:
            return "architect", 4
        elif year <= 2022:
            return "principal", 5
        else:
            return "frontier", 6
    except:
        return "unknown", 0

def extract_repo_with_diffs(repo_path, output_file, max_commits=1000):
    """Extract commits with full diffs from a single repo"""
    repo_path = Path(repo_path)
    repo_name = repo_path.name

    # Detect metadata
    language = detect_language(repo_path)
    frameworks = detect_framework(repo_path)
    domain = detect_domain(repo_name)

    examples = []
    total_tokens = 0

    try:
        # Get commit list
        result = subprocess.run(
            ['git', 'log', '--pretty=format:%H|||%s|||%ai', f'-n{max_commits}'],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return [], 0

        commits = [line for line in result.stdout.strip().split('\n') if line and '|||' in line]

        for line in commits:
            parts = line.split('|||')
            if len(parts) < 3:
                continue

            sha, message, date = parts[0], parts[1], parts[2]
            career_stage, stage_num = get_career_stage(date)

            # Get actual diff (limited size)
            diff_result = subprocess.run(
                ['git', 'show', '--stat', '--patch', '-p', '--no-color', sha],
                cwd=repo_path, capture_output=True, text=True, timeout=60
            )

            if diff_result.returncode != 0:
                continue

            diff_text = diff_result.stdout

            # Limit diff size to avoid huge examples
            if len(diff_text) > 50000:
                diff_text = diff_text[:50000] + "\n... [truncated]"

            # Sanitize the diff
            diff_text = sanitize(diff_text)
            message = sanitize(message)

            # Count tokens (rough estimate)
            tokens = len(diff_text) // 4
            total_tokens += tokens

            # Parse stats from diff
            additions = len(re.findall(r'^\+[^+]', diff_text, re.MULTILINE))
            deletions = len(re.findall(r'^-[^-]', diff_text, re.MULTILINE))

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are Z, a {career_stage.replace('_', ' ')} software developer specializing in {language} and {domain.replace('_', '/')} development."
                    },
                    {
                        "role": "user",
                        "content": f"Implement this change to {repo_name}: {message}"
                    },
                    {
                        "role": "assistant",
                        "content": f"I'll implement this change.\n\n```diff\n{diff_text}\n```"
                    }
                ],
                "metadata": {
                    "sha": sha[:8],
                    "repo": repo_name,
                    "language": language,
                    "frameworks": frameworks,
                    "domain": domain,
                    "additions": additions,
                    "deletions": deletions,
                    "date": date,
                    "career_stage": career_stage,
                    "stage_num": stage_num,
                    "tokens": tokens,
                    "type": "git_diff"
                }
            }
            examples.append(example)

        # Save to file
        if examples:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')

        return examples, total_tokens

    except Exception as e:
        print(f"   Error: {e}")
        return [], 0

def find_all_repos(work_dir):
    """Find all git repositories"""
    work_dir = Path(work_dir)
    repos = []

    for git_dir in work_dir.rglob('.git'):
        if git_dir.is_dir():
            repo_path = git_dir.parent
            # Skip if inside node_modules or vendor
            if 'node_modules' not in str(repo_path) and 'vendor' not in str(repo_path):
                repos.append(repo_path)

    return sorted(repos, key=lambda x: x.name.lower())

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║     ZEN-AGENTIC: INCREMENTAL GIT EXTRACTION                      ║
║     Processing all repos with full diffs                          ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Load progress
    progress = load_progress()
    processed = set(progress["processed_repos"])

    # Find all repos
    work_dir = Path.home() / 'work'
    all_repos = find_all_repos(work_dir)

    print(f"📁 Found {len(all_repos)} repositories")
    print(f"   Already processed: {len(processed)}")
    print(f"   Remaining: {len(all_repos) - len(processed)}")

    # Process each repo
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_count = 0
    batch_size = 50  # Upload to HF every 50 repos

    for i, repo_path in enumerate(all_repos):
        repo_name = repo_path.name

        # Skip if already processed
        if str(repo_path) in processed:
            continue

        print(f"\n[{i+1}/{len(all_repos)}] Processing {repo_name}...")

        output_file = OUTPUT_DIR / f"{repo_name}.jsonl"
        examples, tokens = extract_repo_with_diffs(repo_path, output_file)

        if examples:
            # Update progress
            progress["processed_repos"].append(str(repo_path))
            progress["total_commits"] += len(examples)
            progress["total_tokens"] += tokens

            # Update language stats
            lang = examples[0]["metadata"]["language"]
            progress["by_language"][lang] = progress["by_language"].get(lang, 0) + len(examples)

            # Update domain stats
            domain = examples[0]["metadata"]["domain"]
            progress["by_domain"][domain] = progress["by_domain"].get(domain, 0) + len(examples)

            # Update stage stats
            for ex in examples:
                stage = ex["metadata"]["career_stage"]
                progress["by_stage"][stage] = progress["by_stage"].get(stage, 0) + 1
                progress["total_additions"] = progress.get("total_additions", 0) + ex["metadata"]["additions"]
                progress["total_deletions"] = progress.get("total_deletions", 0) + ex["metadata"]["deletions"]

            print(f"   ✓ {len(examples)} commits, ~{tokens:,} tokens ({lang}, {domain})")

            # Save progress after each repo
            save_progress(progress)
            batch_count += 1
        else:
            # Mark as processed even if empty
            progress["processed_repos"].append(str(repo_path))
            save_progress(progress)
            print(f"   ○ No commits found")

        # Print stats every 50 repos
        if batch_count > 0 and batch_count % batch_size == 0:
            print(f"\n{'='*60}")
            print(f"Progress: {len(progress['processed_repos'])}/{len(all_repos)} repos")
            print(f"Total commits: {progress['total_commits']:,}")
            print(f"Total tokens: {progress['total_tokens']:,}")
            print(f"{'='*60}\n")

    # Final summary
    print(f"\n{'='*70}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total repos: {len(progress['processed_repos'])}")
    print(f"Total commits: {progress['total_commits']:,}")
    print(f"Total tokens: {progress['total_tokens']:,}")
    print(f"Total lines: +{progress.get('total_additions', 0):,} / -{progress.get('total_deletions', 0):,}")
    print(f"\nBy Language:")
    for lang, count in sorted(progress["by_language"].items(), key=lambda x: -x[1]):
        print(f"   {lang}: {count:,}")
    print(f"\nBy Domain:")
    for domain, count in sorted(progress["by_domain"].items(), key=lambda x: -x[1]):
        print(f"   {domain}: {count:,}")
    print(f"\nBy Career Stage:")
    for stage, count in sorted(progress["by_stage"].items(), key=lambda x: -x[1]):
        print(f"   {stage}: {count:,}")

if __name__ == '__main__':
    main()
