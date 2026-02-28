#!/usr/bin/env python3
"""
Robust Git Extraction - Sequential, No Hanging
Skips already processed repos, strict timeouts
"""

import json
import subprocess
import re
import os
import sys
import signal
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / 'data' / 'full_history'
MAX_DIFF = 30000
TIMEOUT = 10  # seconds per git command

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

def sanitize(text):
    if not isinstance(text, str):
        return str(text) if text else ""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'ghp_[a-zA-Z0-9]{36}', '[TOKEN]', text)
    return text

def get_stage(d):
    try:
        y = int(d[:4])
        if y <= 2010: return "early"
        elif y <= 2013: return "growth"
        elif y <= 2016: return "senior"
        elif y <= 2019: return "architect"
        elif y <= 2022: return "principal"
        else: return "frontier"
    except: return "dev"

def extract(repo_path):
    name = f"{repo_path.parent.name}_{repo_path.name}"
    out = OUTPUT_DIR / f"{name}.jsonl"

    if out.exists() and out.stat().st_size > 0:
        return name, 0, 0, "skip"

    examples = []
    bytes_total = 0

    try:
        signal.signal(signal.SIGALRM, timeout_handler)

        # Get commits
        signal.alarm(TIMEOUT)
        r = subprocess.run(
            ['git', 'log', '--pretty=format:%H|||%s|||%ai', '-n2000'],
            cwd=repo_path, capture_output=True, text=True, errors='replace'
        )
        signal.alarm(0)

        if r.returncode != 0:
            return name, 0, 0, "nolog"

        for line in r.stdout.split('\n'):
            if '|||' not in line:
                continue
            parts = line.split('|||')
            if len(parts) < 3:
                continue

            sha, msg, date = parts[0], parts[1], parts[2]

            try:
                signal.alarm(TIMEOUT)
                dr = subprocess.run(
                    ['git', 'show', '--patch', '--no-color', sha],
                    cwd=repo_path, capture_output=True, text=True, errors='replace'
                )
                signal.alarm(0)

                if dr.returncode != 0:
                    continue

                diff = dr.stdout
                if len(diff) > MAX_DIFF:
                    diff = diff[:MAX_DIFF] + "\n..."

                diff = sanitize(diff)
                msg = sanitize(msg)
                stage = get_stage(date)
                bytes_total += len(diff)

                examples.append({
                    "messages": [
                        {"role": "system", "content": f"You are Z, a {stage} developer."},
                        {"role": "user", "content": f"Implement: {msg}"},
                        {"role": "assistant", "content": f"```diff\n{diff}\n```"}
                    ],
                    "metadata": {"sha": sha[:8], "repo": name, "date": date, "stage": stage}
                })

            except TimeoutError:
                continue
            except:
                continue

        if examples:
            with open(out, 'w') as f:
                for e in examples:
                    f.write(json.dumps(e) + '\n')

        return name, len(examples), bytes_total, None

    except TimeoutError:
        return name, 0, 0, "timeout"
    except Exception as e:
        return name, 0, 0, str(e)[:30]
    finally:
        signal.alarm(0)

def find_repos():
    r = subprocess.run(
        f'find {os.path.expanduser("~/work")} {os.path.expanduser("~/play")} -maxdepth 4 -name .git -type d 2>/dev/null | grep -v node_modules | grep -v vendor | grep -v .venv',
        shell=True, capture_output=True, text=True
    )
    repos = []
    for l in r.stdout.split('\n'):
        if l.strip():
            repos.append(Path(l).parent)
    return sorted(set(repos))

def main():
    print("="*70)
    print("ZEN-AGENTIC: ROBUST GIT EXTRACTION")
    print("="*70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    repos = find_repos()
    print(f"Found {len(repos)} repos")

    total_commits = 0
    total_bytes = 0
    done = 0
    skipped = 0

    for i, repo in enumerate(repos):
        name, commits, bytes_c, err = extract(repo)
        done += 1
        if err == "skip":
            skipped += 1
        else:
            total_commits += commits
            total_bytes += bytes_c

        if (i+1) % 20 == 0 or i == len(repos)-1:
            gb = total_bytes / 1024**3
            tokens = total_bytes // 4
            print(f"[{done}/{len(repos)}] Skip:{skipped} Commits:{total_commits:,} Tokens:{tokens/1e9:.2f}B Size:{gb:.2f}GB")

    print(f"\nDONE: {total_commits:,} commits, {total_bytes//4:,} tokens ({total_bytes/1024**3:.2f}GB)")
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump({"repos": done, "commits": total_commits, "tokens": total_bytes//4, "bytes": total_bytes}, f)

if __name__ == '__main__':
    main()
