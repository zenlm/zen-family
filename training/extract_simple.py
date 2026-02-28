#!/usr/bin/env python3
"""Simple extraction with subprocess timeout"""

import json, subprocess, re, os, sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'data' / 'full_history'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sanitize(t):
    if not t: return ""
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[E]', str(t))
    t = re.sub(r'sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16}', '[K]', t)
    return t

def stage(d):
    try:
        y = int(d[:4])
        return "early" if y<=2010 else "growth" if y<=2013 else "senior" if y<=2016 else "arch" if y<=2019 else "principal" if y<=2022 else "frontier"
    except: return "dev"

def extract(repo):
    name = f"{repo.parent.name}_{repo.name}"
    out = OUTPUT_DIR / f"{name}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return name, 0, 0, "skip"

    try:
        r = subprocess.run(['git','log','--pretty=format:%H|||%s|||%ai','-n1000'],
            cwd=repo, capture_output=True, text=True, timeout=15, errors='replace')
        if r.returncode != 0: return name, 0, 0, "nolog"

        examples, total = [], 0
        for line in r.stdout.split('\n')[:500]:  # Limit commits
            if '|||' not in line: continue
            p = line.split('|||')
            if len(p) < 3: continue
            sha, msg, date = p[0], p[1], p[2]

            try:
                dr = subprocess.run(['git','show','--patch','--no-color',sha],
                    cwd=repo, capture_output=True, text=True, timeout=10, errors='replace')
                if dr.returncode != 0: continue
                diff = dr.stdout[:20000]
                diff, msg = sanitize(diff), sanitize(msg)
                total += len(diff)
                examples.append({
                    "messages": [
                        {"role":"system","content":f"You are Z, a {stage(date)} developer."},
                        {"role":"user","content":f"Implement: {msg}"},
                        {"role":"assistant","content":f"```diff\n{diff}\n```"}
                    ],
                    "metadata": {"sha":sha[:8],"repo":name,"date":date,"stage":stage(date)}
                })
            except subprocess.TimeoutExpired: continue
            except: continue

        if examples:
            with open(out,'w') as f:
                for e in examples: f.write(json.dumps(e)+'\n')
        return name, len(examples), total, None

    except subprocess.TimeoutExpired: return name, 0, 0, "timeout"
    except Exception as e: return name, 0, 0, str(e)[:20]

def main():
    print("EXTRACTION STARTING...")
    r = subprocess.run(
        f'find {os.path.expanduser("~/work")} {os.path.expanduser("~/play")} -maxdepth 4 -name .git -type d 2>/dev/null',
        shell=True, capture_output=True, text=True)
    repos = [Path(l).parent for l in r.stdout.split('\n') if l.strip() and 'node_modules' not in l]
    repos = sorted(set(repos))
    print(f"Found {len(repos)} repos")

    tc, tb, done, skip = 0, 0, 0, 0
    for i, repo in enumerate(repos):
        n, c, b, e = extract(repo)
        done += 1
        if e == "skip": skip += 1
        else: tc += c; tb += b
        if (i+1) % 10 == 0:
            print(f"[{done}/{len(repos)}] Skip:{skip} Commits:{tc:,} Tokens:{tb//4/1e6:.1f}M Size:{tb/1e9:.2f}GB")
            sys.stdout.flush()

    print(f"\nDONE: {tc:,} commits, {tb//4/1e9:.2f}B tokens")
    with open(OUTPUT_DIR/'summary.json','w') as f:
        json.dump({"repos":done,"commits":tc,"tokens":tb//4,"bytes":tb},f)

if __name__ == '__main__':
    main()
