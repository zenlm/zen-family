#!/usr/bin/env python3
"""
Create placeholder PDFs for all Zen model whitepapers
This creates simple PDFs with the whitepaper content
"""

import os
from pathlib import Path

def create_placeholder_pdfs():
    """Create placeholder PDF files with basic info"""
    
    latex_dir = Path("/Users/z/work/zen/docs/papers/latex")
    pdf_dir = Path("/Users/z/work/zen/docs/papers/pdf")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    papers = [
        ("zen_family_overview", "Zen AI Model Family - Technical Overview", "19.4KB"),
        ("zen-nano_whitepaper", "Zen-Nano 0.6B - Technical Whitepaper", "8.8KB"),
        ("zen-eco_whitepaper", "Zen-Eco 4B - Technical Whitepaper", "8.9KB"),
        ("zen-omni_whitepaper", "Zen-Omni 30B - Technical Whitepaper", "9.1KB"),
        ("zen-coder_whitepaper", "Zen-Coder 480B - Technical Whitepaper", "9.3KB"),
        ("zen-next_whitepaper", "Zen-Next 80B - Technical Whitepaper", "9.0KB"),
        ("zen-artist_whitepaper", "Zen-Artist 8B - Technical Whitepaper", "8.8KB"),
        ("zen-artist-edit_whitepaper", "Zen-Artist-Edit 7B - Technical Whitepaper", "8.9KB"),
        ("zen-designer-thinking_whitepaper", "Zen-Designer-Thinking 235B - Technical Whitepaper", "9.3KB"),
        ("zen-designer-instruct_whitepaper", "Zen-Designer-Instruct 235B - Technical Whitepaper", "9.2KB"),
        ("zen-scribe_whitepaper", "Zen-Scribe 1.5B - Technical Whitepaper", "8.7KB")
    ]
    
    print("📄 Creating PDF placeholders for Zen whitepapers")
    print("=" * 50)
    
    for filename, title, size in papers:
        # Check if LaTeX file exists
        latex_file = latex_dir / f"{filename}.tex"
        if latex_file.exists():
            # Create a placeholder file indicating PDF will be generated
            pdf_file = pdf_dir / f"{filename}.pdf"
            
            # Create a simple text file as placeholder (rename to .pdf.txt for clarity)
            placeholder_file = pdf_dir / f"{filename}.pdf.placeholder"
            with open(placeholder_file, 'w') as f:
                f.write(f"PDF PLACEHOLDER: {title}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"This is a placeholder for the PDF version of:\n")
                f.write(f"{title}\n\n")
                f.write(f"LaTeX source: docs/papers/latex/{filename}.tex\n")
                f.write(f"Expected size: ~{size}\n\n")
                f.write("To generate the actual PDF, run:\n")
                f.write(f"  pdflatex docs/papers/latex/{filename}.tex\n\n")
                f.write("Or use the script:\n")
                f.write("  ./scripts/generate_all_pdfs.sh\n")
            
            print(f"✅ Created placeholder: {filename}.pdf.placeholder")
    
    print("\n" + "=" * 50)
    print("✅ Placeholder creation complete!")
    print(f"📁 Location: {pdf_dir}")
    print("\n💡 To generate actual PDFs, install LaTeX and run:")
    print("   ./scripts/generate_all_pdfs.sh")

if __name__ == "__main__":
    create_placeholder_pdfs()