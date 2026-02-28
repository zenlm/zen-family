#!/usr/bin/env python3
"""
Convert Claude Code JSONL logs to training dataset format
Sanitizes PII and prepares for model training
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Sanitization patterns
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'api_key': r'(sk-[a-zA-Z0-9]{48}|xox[baprs]-[a-zA-Z0-9-]+)',
    'path': r'/Users/[^/\s]+',
    'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

def sanitize_text(text: str) -> str:
    """Remove PII from text"""
    if not text:
        return text
    
    # Replace emails
    text = re.sub(PII_PATTERNS['email'], '[EMAIL]', text)
    
    # Replace API keys
    text = re.sub(PII_PATTERNS['api_key'], '[API_KEY]', text)
    
    # Replace user paths
    text = re.sub(PII_PATTERNS['path'], '/Users/[USER]', text)
    
    # Replace IPs
    text = re.sub(PII_PATTERNS['ip'], '[IP]', text)
    
    return text

def process_claude_jsonl(input_path: str) -> List[Dict]:
    """Process Claude Code JSONL file"""
    conversations = []
    
    try:
        with open(input_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract conversation data
                    if data.get('type') == 'message':
                        message = {
                            'timestamp': data.get('timestamp'),
                            'content': sanitize_text(data.get('display', '')),
                            'type': data.get('type'),
                        }
                        
                        # Add usage data if available
                        if 'usage' in data:
                            message['tokens'] = data['usage']
                        
                        conversations.append(message)
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    
    return conversations

def convert_to_training_format(conversations: List[Dict]) -> List[Dict]:
    """Convert to training dataset format"""
    training_data = []
    
    # Group into conversation pairs
    for i in range(0, len(conversations) - 1, 2):
        if i + 1 < len(conversations):
            entry = {
                'messages': [
                    {'role': 'user', 'content': conversations[i]['content']},
                    {'role': 'assistant', 'content': conversations[i+1]['content']}
                ],
                'metadata': {
                    'timestamp': conversations[i]['timestamp'],
                    'tokens': conversations[i+1].get('tokens', {})
                }
            }
            training_data.append(entry)
    
    return training_data

def main():
    print("🧹 Processing Claude Code logs for training dataset")
    print("=" * 60)
    
    claude_dir = Path.home() / '.claude' / 'projects'
    output_dir = Path.home() / 'work' / 'zen' / 'zen-family' / 'training' / 'data'
    output_file = output_dir / f'claude_sanitized_{datetime.now().strftime("%Y%m%d")}.jsonl'
    
    all_training_data = []
    total_tokens = 0
    files_processed = 0
    
    print(f"\n📂 Scanning: {claude_dir}")
    
    # Process all JSONL files
    for jsonl_file in claude_dir.rglob('*.jsonl'):
        conversations = process_claude_jsonl(str(jsonl_file))
        if conversations:
            training_examples = convert_to_training_format(conversations)
            all_training_data.extend(training_examples)
            
            # Count tokens
            for example in training_examples:
                tokens = example['metadata'].get('tokens', {})
                total_tokens += tokens.get('total', 0)
            
            files_processed += 1
            if files_processed % 100 == 0:
                print(f"  Processed: {files_processed} files...")
    
    print(f"\n✅ Processed {files_processed} files")
    print(f"✅ Extracted {len(all_training_data)} training examples")
    print(f"✅ Total tokens: {total_tokens:,}")
    
    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in all_training_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Examples: {len(all_training_data)}")
    
    print("\n✅ Dataset ready for training!")

if __name__ == '__main__':
    main()
