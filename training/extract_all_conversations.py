#!/usr/bin/env python3
"""
Extract ALL conversations from Claude Code sessions
Properly handles the full 18B+ tokens
"""

import json
from pathlib import Path
from tqdm import tqdm
import re

def sanitize(text):
    """Remove PII"""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'sk-[a-zA-Z0-9]{48}', '[API_KEY]', text)
    text = re.sub(r'/Users/[^/\s]+', '/Users/[USER]', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    text = re.sub(r'oauth[_-]token[_-]?secret?["\']?\s*[:=]\s*["\']?[a-f0-9:]+', 'oauth_token=[REDACTED]', text)
    return text

def extract_text(content):
    """Extract text from content (string or array)"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    texts.append(item.get('text', ''))
                elif 'text' in item:
                    texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        return '\n'.join(filter(None, texts))
    return str(content)

def process_file(path):
    """Process one JSONL file and extract ALL conversations"""
    conversations = []
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Only process user/assistant messages (not summaries, snapshots, etc)
                if data.get('type') not in ['user', 'assistant']:
                    continue
                
                msg = data.get('message', {})
                if not msg:
                    continue
                
                # Extract content
                content = extract_text(msg.get('content', ''))
                if not content or len(content) < 10:  # Skip tiny messages
                    continue
                
                # Get usage (might be in message.usage or top-level)
                usage = msg.get('usage') or data.get('usage') or {}
                
                conversations.append({
                    'role': data['type'],
                    'content': sanitize(content),
                    'timestamp': data.get('timestamp'),
                    'tokens': {
                        'input': usage.get('input_tokens', 0),
                        'output': usage.get('output_tokens', 0),
                        'cache_creation': usage.get('cache_creation_input_tokens', 0),
                        'cache_read': usage.get('cache_read_input_tokens', 0)
                    }
                })
                
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
                
    except Exception:
        pass
    
    return conversations

def main():
    print("🚀 Extracting ALL conversations from Claude Code")
    print("=" * 70)
    
    claude_dir = Path.home() / '.claude' / 'projects'
    output_file = Path.home() / 'work' / 'zen' / 'zen-family' / 'training' / 'data' / 'claude_all_conversations.jsonl'
    
    # Find all JSONL
    files = list(claude_dir.rglob('*.jsonl'))
    print(f"\nProcessing {len(files)} files...")
    
    all_conversations = []
    total_tokens = 0
    
    for file_path in tqdm(files, desc="Extracting"):
        convos = process_file(file_path)
        all_conversations.extend(convos)
        
        for c in convos:
            t = c['tokens']
            total_tokens += t['input'] + t['output'] + t['cache_creation'] + t['cache_read']
    
    print(f"\n✅ Extracted:")
    print(f"   Total messages: {len(all_conversations):,}")
    print(f"   Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    
    # Create training pairs
    print(f"\n📝 Creating training examples...")
    examples = []
    
    i = 0
    while i < len(all_conversations) - 1:
        if all_conversations[i]['role'] == 'user' and all_conversations[i+1]['role'] == 'assistant':
            examples.append({
                'messages': [
                    {'role': 'user', 'content': all_conversations[i]['content']},
                    {'role': 'assistant', 'content': all_conversations[i+1]['content']}
                ],
                'metadata': {
                    'timestamp': all_conversations[i]['timestamp'],
                    'tokens': all_conversations[i+1]['tokens']
                }
            })
            i += 2
        else:
            i += 1
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    size_mb = output_file.stat().st_size / 1024 / 1024
    
    print(f"\n💾 Dataset saved:")
    print(f"   File: {output_file.name}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Examples: {len(examples):,}")
    print(f"   Tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"\n✅ Ready for training!")

if __name__ == '__main__':
    main()
