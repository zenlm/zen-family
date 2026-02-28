# How to Find Claude Code Data on Any Computer

## Location of Claude Code Session Data

### Primary Location
```bash
~/.claude/projects/
```

This directory contains ALL your Claude Code conversation history in JSONL format.

### Structure
```
~/.claude/
├── projects/              # Main conversation data
│   ├── -Users-{user}-{project}/
│   │   ├── {session-uuid}.jsonl     # Full conversations (user + assistant)
│   │   └── agent-{id}.jsonl         # Sub-agent conversations
│   └── ...
├── file-history/         # File edit history
├── debug/                # Debug logs
└── history.jsonl         # Current session summary
```

## How to Audit Total Tokens

Run this command to see ALL your tokens:

```bash
cd ~/.claude
node audit_all_tokens.js
```

This will scan all 1,400+ JSONL files and show you the total (should be 10B+ if you've been coding a lot).

## Expected Token Counts

For heavy Claude Code usage:
- **Daily**: 100M - 1B tokens
- **Weekly**: 1B - 5B tokens  
- **Monthly**: 5B - 20B tokens

If your audit shows less, check:
1. Other user accounts on the machine
2. Old ~/.config/claude directories
3. Archived sessions

## JSONL File Format

Each line in the JSONL files is a JSON object:

```json
{
  "type": "assistant",
  "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "..."}],
    "usage": {
      "input_tokens": 1000,
      "output_tokens": 500,
      "cache_creation_input_tokens": 50000,
      "cache_read_input_tokens": 10000
    }
  },
  "timestamp": "2025-11-13T...",
  "uuid": "..."
}
```

## On Other Computers

### 1. Check Default Locations

```bash
# Primary location
ls -la ~/.claude/projects

# Alternative location  
ls -la ~/.config/claude/projects

# Check both user accounts
ls -la /Users/*/. claude/projects
```

### 2. Search for JSONL Files

```bash
# Find all Claude data
find ~ -name "*.jsonl" -path "*/.claude/*" 2>/dev/null

# Count total
find ~ -name "*.jsonl" -path "*/.claude/*" 2>/dev/null | wc -l
```

### 3. Check Disk Usage

```bash
# See how much space Claude data uses
du -sh ~/.claude/projects
du -sh ~/.config/claude 2>/dev/null
```

### 4. Find Largest Sessions

```bash
# See biggest conversation files
find ~/.claude/projects -name "*.jsonl" -exec wc -l {} \; | sort -rn | head -20
```

## Multi-User Setup

If you have multiple users (like @hanzoai and @zeekay):

```bash
# Check both home directories
sudo ls -la /Users/z/.claude/projects
sudo ls -la /Users/hanzo/.claude/projects

# Or check all users
for user in /Users/*; do
  if [ -d "$user/.claude/projects" ]; then
    echo "Found Claude data for: $(basename $user)"
    du -sh "$user/.claude/projects"
  fi
done
```

## Syncing Across Computers

### Method 1: Git (for sanitized data only)
```bash
# On computer A
cd ~/.claude/projects
git init
git add *.jsonl
git commit -m "Claude sessions"
git push

# On computer B  
git clone your-repo ~/.claude/projects-backup
```

### Method 2: rsync
```bash
# From computer A to B
rsync -avz ~/.claude/projects/ user@computerB:~/.claude/projects/
```

### Method 3: Cloud Sync
```bash
# Using Dropbox/Google Drive
ln -s ~/Dropbox/claude-backups ~/.claude/projects-backup
cp -r ~/.claude/projects/* ~/Dropbox/claude-backups/
```

## Finding Data After Reinstall

If you reinstalled Claude Code:

```bash
# Check Time Machine backups
tmutil listbackups
tmutil restore /path/to/backup/.claude ~/.claude-recovered

# Check iCloud
ls -la ~/Library/Mobile\ Documents/com~apple~CloudDocs/.claude

# Check any backup drives
find /Volumes -name ".claude" 2>/dev/null
```

## Expected File Sizes

For reference:
- Small session (< 1 hour): 10-100 KB
- Medium session (few hours): 1-10 MB  
- Large session (full day): 10-100 MB
- Huge session (week-long): 100MB - 1GB

If you have 10B+ tokens, expect:
- Total .jsonl files: ~10-50 GB
- Number of files: 1,000 - 5,000
- Largest files: 50-200 MB each

## Security Note

⚠️ **NEVER share raw .jsonl files** - they contain:
- API keys
- File paths
- Project names
- Personal information

**ALWAYS sanitize first** using:
```bash
cd ~/work/zen/zen-family/training
python3 process_claude_logs_v2.py
```

This removes all PII before creating training datasets.

## Troubleshooting

**"Can't find any data"**
- Check you're looking in the right user directory
- Claude Code might be in ~/.config/claude instead
- Check if data was archived/moved

**"Files exist but show 0 tokens"**
- Files might be summaries not full conversations
- Check file sizes - should be > 1MB for real sessions
- Look for files with "usage" field in JSON

**"Total is way less than expected"**
- Check if old sessions were deleted
- Look for backup/archive directories
- Check Time Machine or other backups

## Automated Backup Script

```bash
#!/bin/bash
# Save to ~/backup-claude.sh
DATE=$(date +%Y%m%d)
tar -czf ~/claude-backup-$DATE.tar.gz ~/.claude/projects
echo "Backed up to: ~/claude-backup-$DATE.tar.gz"
```

Run daily via cron:
```bash
0 2 * * * ~/backup-claude.sh
```

---

For questions: Check ~/work/zen/zen-family/training/process_claude_logs_v2.py