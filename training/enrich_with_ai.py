#!/usr/bin/env python3
"""
Enrich Training Data with AI (Haiku)

Adds rich context to each commit:
- Detailed explanations of changes
- Organization/project context
- Technical stack info
- Time period context
- Better formatted training examples
"""

import json
import os
import anthropic
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

INPUT_DIR = Path(__file__).parent / 'data' / 'full_history'
OUTPUT_DIR = Path(__file__).parent / 'data' / 'enriched'
BATCH_SIZE = 100

client = anthropic.Anthropic()

# Organization context
ORG_CONTEXT = {
    "lux": "Lux Network - Post-quantum blockchain with novel consensus",
    "hanzo": "Hanzo AI - Frontier AI infrastructure and LLM tooling",
    "zoo": "Zoo Labs - Decentralized AI research network",
    "zeekay": "Personal projects and experiments",
}

def get_org(repo_name):
    """Get organization from repo name"""
    for org, desc in ORG_CONTEXT.items():
        if org in repo_name.lower():
            return org, desc
    return "general", "General software development"

def enrich_example(example):
    """Add AI-generated context to a training example"""
    try:
        metadata = example.get("metadata", {})
        repo = metadata.get("repo", "unknown")
        date = metadata.get("date", "")
        stage = metadata.get("stage", "developer")

        # Get organization context
        org, org_desc = get_org(repo)

        # Get the diff content
        assistant_msg = example["messages"][-1]["content"]
        user_msg = example["messages"][1]["content"]

        # Use Haiku to generate enriched context
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Summarize this git commit in 2-3 sentences. Focus on what was changed and why.

Repository: {repo}
Organization: {org_desc}
Date: {date}
Commit: {user_msg}

Diff preview (first 2000 chars):
{assistant_msg[:2000]}

Respond with just the summary, no preamble."""
            }]
        )

        summary = response.content[0].text.strip()

        # Create enriched example
        enriched = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are Z, a {stage.replace('_', ' ')} developer at {org_desc}. You write clean, efficient code with deep expertise in blockchain, AI, and full-stack development."
                },
                {
                    "role": "user",
                    "content": user_msg
                },
                {
                    "role": "assistant",
                    "content": f"{summary}\n\n{assistant_msg}"
                }
            ],
            "metadata": {
                **metadata,
                "organization": org,
                "org_description": org_desc,
                "ai_summary": summary,
                "enriched": True
            }
        }

        return enriched, None

    except Exception as e:
        return example, str(e)

def process_file(input_file):
    """Process a single file and enrich all examples"""
    output_file = OUTPUT_DIR / input_file.name

    if output_file.exists():
        return input_file.name, 0, "exists"

    examples = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if not examples:
        return input_file.name, 0, "empty"

    enriched = []
    errors = 0

    for ex in examples:
        result, error = enrich_example(ex)
        enriched.append(result)
        if error:
            errors += 1
        time.sleep(0.1)  # Rate limit

    with open(output_file, 'w') as f:
        for ex in enriched:
            f.write(json.dumps(ex) + '\n')

    return input_file.name, len(enriched), f"{errors} errors" if errors else None

def main():
    print("="*70)
    print("ZEN-AGENTIC: AI ENRICHMENT (Haiku)")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all input files
    files = list(INPUT_DIR.glob("*.jsonl"))
    print(f"\n📁 Found {len(files)} files to enrich")

    total_enriched = 0

    for i, f in enumerate(files):
        name, count, error = process_file(f)
        total_enriched += count

        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{len(files)}] Enriched: {total_enriched:,}")

    print(f"\n✅ COMPLETE: {total_enriched:,} examples enriched")
    print(f"   Output: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
