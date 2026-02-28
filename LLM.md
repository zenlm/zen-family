# zen-family — Technical Whitepapers

This repository contains LaTeX whitepapers for the Zen AI model family.

## Papers

Located in `docs/papers/latex/`:

| File | Model | Description |
|------|-------|-------------|
| `zen_family_overview.tex` | All | Family overview and architecture |
| `zen-nano_whitepaper.tex` | Zen-Nano (0.6B) | Edge deployment model |
| `zen-eco_whitepaper.tex` | Zen-Eco (4B) | Consumer hardware model |
| `zen-omni_whitepaper.tex` | Zen-Omni (30B MoE) | Multimodal model |
| `zen-coder_whitepaper.tex` | Zen-Coder (various) | Code generation family |
| `zen-next_whitepaper.tex` | Zen-Next (32B) | General reasoning |
| `zen-artist_whitepaper.tex` | Zen-Artist (8B) | Text-to-image |
| `zen-artist-edit_whitepaper.tex` | Zen-Artist-Edit (7B) | Image editing |
| `zen-designer-instruct_whitepaper.tex` | Zen-Designer (235B) | Visual understanding |
| `zen-designer-thinking_whitepaper.tex` | Zen-Designer-Think (235B) | Visual reasoning |
| `zen-scribe_whitepaper.tex` | Zen-Scribe (1.5B) | Speech recognition |
| `zen-guard_whitepaper.tex` | Zen-Guard | Safety model |

## Build PDFs

```bash
cd docs/papers/latex
for tex in *.tex; do pdflatex "$tex"; done
```

## Brand Policy

All papers must use Zen MoDE (Mixture of Distilled Experts) branding. Never reference upstream model names. See `~/work/hanzo/CLAUDE.md` for full brand policy.

## SOTA References (2026)

Key papers to cite in technical sections:
- **BitDelta** (arXiv:2402.10193) — 1-bit delta compression for multi-variant serving
- **SuRe** (arXiv:2511.22367) — Surprise-driven replay for continual learning
- **OPCM** (arXiv:2501.09522) — Sequential continual model merging
- **OPLoRA** (arXiv:2510.13003) — Orthogonal projection LoRA for forgetting prevention
- **Drop-Upcycling** (arXiv:2502.19261) — Dense-to-MoE conversion with partial re-init
- **MonoSoup** (arXiv:2602.09689) — Single-checkpoint SVD merging
- **Q-GaLore** (2024) — Memory-efficient fine-tuning with quantized gradient projections
- **Youtu-Agent** (arXiv:2512.24615) — Training-free GRPO via in-context accumulation
