#!/bin/bash

# Generate all PDFs from LaTeX whitepapers
echo "📚 GENERATING ZEN MODEL PDFs"
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
    
    # Run pdflatex twice to resolve references
    if pdflatex -interaction=nonstopmode -output-directory=. "$tex" > /dev/null 2>&1 && \
       pdflatex -interaction=nonstopmode -output-directory=. "$tex" > /dev/null 2>&1; then
        # Move PDF to pdfs directory
        if [ -f "${basename}.pdf" ]; then
            mv "${basename}.pdf" "../pdfs/"
            echo "  ✅ Generated: ${basename}.pdf"
            success=$((success + 1))
        fi
        
        # Clean up auxiliary files
        rm -f "${basename}.aux" "${basename}.log" "${basename}.out" 2>/dev/null
    else
        echo "  ❌ Failed to generate PDF for $basename"
    fi
done

echo ""
echo "============================================================"
echo "✅ COMPLETED: Generated $success/$total PDFs"
echo ""
echo "📁 PDFs location: /Users/z/work/zen/docs/papers/pdfs/"
echo ""
ls -lh ../pdfs/*.pdf 2>/dev/null

# Also create a combined markdown with links
echo ""
echo "📝 Creating markdown index..."
cat > ../pdfs/README.md << 'EOF'
# Zen AI Model Family - Technical Whitepapers

## Complete Model Family
- [Zen Family Overview](./zen_family_overview.pdf) - Complete ecosystem documentation

## Language Models
- [Zen-Nano 0.6B](./zen-nano_whitepaper.pdf) - Ultra-efficient nano model
- [Zen-Eco 4B](./zen-eco_whitepaper.pdf) - Balanced efficiency model
- [Zen-Omni 30B](./zen-omni_whitepaper.pdf) - Versatile general-purpose model
- [Zen-Coder 480B MoE](./zen-coder_whitepaper.pdf) - Advanced code generation
- [Zen-Next 80B](./zen-next_whitepaper.pdf) - Next-generation capabilities

## Multimodal Models
- [Zen-Artist](./zen-artist_whitepaper.pdf) - Text-to-image generation
- [Zen-Artist-Edit](./zen-artist-edit_whitepaper.pdf) - Advanced image editing
- [Zen-Designer Thinking 235B MoE](./zen-designer-thinking_whitepaper.pdf) - Visual reasoning with thinking mode
- [Zen-Designer Instruct 235B MoE](./zen-designer-instruct_whitepaper.pdf) - Vision-language understanding
- [Zen-Scribe](./zen-scribe_whitepaper.pdf) - Speech recognition and transcription

## Safety & Moderation
- [Zen-Guard](./zen-guard_whitepaper.pdf) - Content safety and moderation

---

*Generated: $(date)*
EOF

echo "✅ Created markdown index: ../pdfs/README.md"