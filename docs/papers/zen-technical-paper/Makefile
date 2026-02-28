# Makefile for Zen Technical Paper

PAPER = main
LATEX = pdflatex
BIBTEX = bibtex
RM = rm -f

# Default target
all: $(PAPER).pdf

# Build the PDF
$(PAPER).pdf: $(PAPER).tex sections/*.tex
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER) || true
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

# Clean intermediate files
clean:
	$(RM) $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg
	$(RM) $(PAPER).out $(PAPER).toc $(PAPER).lot $(PAPER).lof
	$(RM) $(PAPER).nav $(PAPER).snm $(PAPER).vrb
	$(RM) *.aux *.log *.bbl *.blg *.out *.toc

# Clean everything including PDF
distclean: clean
	$(RM) $(PAPER).pdf

# View the PDF
view: $(PAPER).pdf
	open $(PAPER).pdf || xdg-open $(PAPER).pdf

# Quick build without bibliography
quick:
	$(LATEX) $(PAPER)

.PHONY: all clean distclean view quick