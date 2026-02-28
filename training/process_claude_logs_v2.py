#!/usr/bin/env python3
"""
Convert Claude Code JSONL logs to training dataset
Processes 18B+ tokens from ~/.claude/projects
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def sanitize_text(text: str) -> str:
    """Remove PII"""
    if not isinstance(text, str):
        return str(text)
    
    # Patterns
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'sk-[a-zA-Z0-9]{48}', '[API_KEY]', text)
    text = re.sub(r'/Users/[^/\s]+', '/Users/[USER]', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    text = re.sub(r'(oauth_token|oauth_token_secret)["\']?\s*[:=]\s*["\']?[a-f0-9:]+', r'\1=[REDACTED]', text)
    
    return text

def extract_content(message):
    """Extract text content from message"""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if 'content' in message:
            content = message['content']
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
                return '\n'.join(texts)
        return message.get('text', str(message))
    return str(message)

def process_jsonl_file(file_path):
    """Process a single JSONL file"""
    messages = []
    total_tokens = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Look for conversation messages
                    if data.get('type') in ['user', 'assistant']:
                        msg = data.get('message', {})
                        
                        content = extract_content(msg)
                        if content:
                            messages.append({
                                'role': data['type'],
                                'content': sanitize_text(content),
                                'timestamp': data.get('timestamp'),
                                'usage': data.get('usage', {})
                            })
                            
                            # Count tokens
                            usage = data.get('usage', {})
                            if usage:
                                total_tokens += usage.get('input_tokens', 0)
                                total_tokens += usage.get('output_tokens', 0)
                                total_tokens += usage.get('cache_creation_input_tokens', 0)
                                total_tokens += usage.get('cache_read_input_tokens', 0)
                                
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        return [], 0
    
    return messages, total_tokens

def create_training_examples(messages):
    """Convert to training format"""
    examples = []
    
    i = 0
    while i < len(messages) - 1:
        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
            example = {
                'messages': [
                    {'role': 'user', 'content': messages[i]['content']},
                    {'role': 'assistant', 'content': messages[i+1]['content']}
                ],
                'metadata': {
                    'timestamp': messages[i]['timestamp'],
                    'tokens': messages[i+1].get('usage', {})
                }
            }
            examples.append(example)
            i += 2
        else:
            i += 1
    
    return examples

def main():
    print("🧹 Processing 18B+ tokens from Claude Code logs")
    print("=" * 70)
    
    claude_dir = Path.home() / '.claude' / 'projects'
    output_dir = Path.home() / 'work' / 'zen' / 'zen-family' / 'training' / 'data'
    output_file = output_dir / f'claude_code_training_{datetime.now().strftime("%Y%m%d_%H%M")}.jsonl'
    
    print(f"\n📂 Input: {claude_dir}")
    print(f"📂 Output: {output_file}")
    print()
    
    # Find all JSONL files
    jsonl_files = list(claude_dir.rglob('*.jsonl'))
    print(f"Found {len(jsonl_files)} JSONL files\n")
    
    all_examples = []
    grand_total_tokens = 0
    files_with_data = 0
    
    # Process with progress bar
    for file_path in tqdm(jsonl_files, desc="Processing files"):
        messages, tokens = process_jsonl_file(file_path)
        
        if messages:
            examples = create_training_examples(messages)
            all_examples.extend(examples)
            grand_total_tokens += tokens
            files_with_data += 1
    
    print(f"\n✅ Results:")
    print(f"   Files with data: {files_with_data}/{len(jsonl_files)}")
    print(f"   Training examples: {len(all_examples):,}")
    print(f"   Total tokens: {grand_total_tokens:,} ({grand_total_tokens/1e9:.2f}B)")
    
    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\n💾 Saved dataset:")
    print(f"   File: {output_file.name}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Examples: {len(all_examples):,}")
    print(f"\n✅ Dataset ready for training!")

if __name__ == '__main__':
    main()
