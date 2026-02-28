#!/bin/bash
# Zen Family GitHub Upload Script

echo "🚀 Uploading Zen Family to GitHub..."

# Add all new files
git add docs/
git add ZEN_FAMILY.md
git add README.md
git add complete_zen_family_setup.py

# Commit changes
git commit -m "feat: Complete Zen AI Model Family with 10 models and documentation

- Added Zen-Artist (text-to-image) and Zen-Artist-Edit (image editing)
- Added Zen-Scribe (ASR/speech recognition)
- Created comprehensive LaTeX whitepapers for all 10 models
- Generated family overview documentation
- Updated README with complete model lineup
- Added technical papers in LaTeX and PDF formats
- Structured documentation in docs/papers/
- Linked all HuggingFace repositories"

# Push to GitHub
git push origin main

echo "✅ Successfully uploaded Zen Family to GitHub!"
echo "📚 View at: https://github.com/zenlm/zen"
