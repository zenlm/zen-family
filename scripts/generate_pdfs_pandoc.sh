#!/bin/bash

# Generate all PDFs from LaTeX whitepapers using pandoc
echo "📚 GENERATING ZEN MODEL PDFs (using pandoc)"
echo "============================================================"

cd /Users/z/work/zen/docs/papers/latex

# Create PDF output directory
mkdir -p ../pdfs

# Counter for tracking progress
total=0
success=0

# Process each LaTeX file
for tex in *.tex; do
    total=$((total + 1))
    basename="${tex%.tex}"
    echo ""
    echo "📄 Processing: $basename"
    
    # Use pandoc to convert LaTeX to PDF
    if pandoc "$tex" \
        --pdf-engine=xelatex \
        -o "../pdfs/${basename}.pdf" \
        --variable geometry:margin=1in \
        --variable fontsize=11pt \
        --variable documentclass=article \
        --highlight-style=tango \
        2>/dev/null; then
        echo "  ✅ Generated: ${basename}.pdf"
        success=$((success + 1))
    else
        # Fallback to HTML if PDF fails
        echo "  ⚠️  PDF failed, trying HTML fallback..."
        if pandoc "$tex" \
            -s \
            -o "../pdfs/${basename}.html" \
            --metadata title="${basename}" \
            --css=https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css \
            2>/dev/null; then
            echo "  ✅ Generated HTML: ${basename}.html"
            success=$((success + 1))
        else
            echo "  ❌ Failed to generate output for $basename"
        fi
    fi
done

echo ""
echo "============================================================"
echo "✅ COMPLETED: Generated $success/$total documents"
echo ""
echo "📁 Output location: /Users/z/work/zen/docs/papers/pdfs/"
echo ""
ls -lh ../pdfs/* 2>/dev/null | head -20

# Create markdown index
cat > ../pdfs/README.md << 'EOF'
# Zen AI Model Family - Technical Whitepapers

## 🌟 Complete Ecosystem (11 Models)

### Overview
- [Zen Family Overview](./zen_family_overview.pdf) - Complete ecosystem documentation

### Language Models (5 models)
- [Zen-Nano 0.6B](./zen-nano_whitepaper.pdf) - Ultra-efficient nano model
- [Zen-Eco 4B](./zen-eco_whitepaper.pdf) - Balanced efficiency model  
- [Zen-Omni 30B](./zen-omni_whitepaper.pdf) - Versatile general-purpose model
- [Zen-Coder 480B MoE](./zen-coder_whitepaper.pdf) - Advanced code generation (30B active)
- [Zen-Next 80B](./zen-next_whitepaper.pdf) - Next-generation capabilities

### Multimodal Models (5 models)
- [Zen-Artist](./zen-artist_whitepaper.pdf) - Text-to-image generation
- [Zen-Artist-Edit 7B](./zen-artist-edit_whitepaper.pdf) - Advanced image editing
- [Zen-Designer Thinking 235B MoE](./zen-designer-thinking_whitepaper.pdf) - Visual reasoning (22B active)
- [Zen-Designer Instruct 235B MoE](./zen-designer-instruct_whitepaper.pdf) - Vision-language (22B active)
- [Zen-Scribe](./zen-scribe_whitepaper.pdf) - Speech recognition and transcription

### Safety & Moderation (1 model, 2 variants)
- [Zen-Guard](./zen-guard_whitepaper.pdf) - Content safety and moderation
  - Zen-Guard-Gen-8B: Generative safety classification
  - Zen-Guard-Stream-4B: Real-time token monitoring

## Key Features
- **v1.0.1 Release**: Recursive self-improvement through RAIS
- **94% Effectiveness**: Proven across training examples
- **Multi-format**: GGUF, MLX, SafeTensors support
- **Zoo-Gym Integration**: Advanced training framework
- **119 Languages**: Comprehensive multilingual support
- **Open Source**: Apache 2.0 license

## Partnership
Built by **Hanzo AI** (Techstars-backed) and **Zoo Labs Foundation** (501(c)(3) non-profit)

---

*Generated: $(date)*
*Total Models: 11 (5 Language, 5 Multimodal, 1 Safety)*
EOF

echo "✅ Created markdown index: ../pdfs/README.md"