#!/usr/bin/env bash
set -euo pipefail

# Compile helper for reproducible builds of paper/main.pdf
# Usage:
#   ./compile.sh           # build PDF
#   ./compile.sh pdf       # same as above
#   ./compile.sh clean     # remove aux files
#   ./compile.sh clobber   # remove aux + PDF

cd "$(dirname "$0")"
target="${1:-pdf}"

case "$target" in
  pdf)
    if command -v latexmk >/dev/null 2>&1; then
      latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    else
      echo "latexmk not found; falling back to manual pdflatex/bibtex passes" >&2
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
      # Run bibtex if available to resolve bibliography
      if command -v bibtex >/dev/null 2>&1; then bibtex main || true; fi
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    fi
    ;;
  clean)
    if command -v latexmk >/dev/null 2>&1; then
      latexmk -c || true
    fi
    rm -f main.{aux,bbl,blg,fls,fdb_latexmk,log,lof,lot,out,toc} || true
    ;;
  clobber)
    if command -v latexmk >/dev/null 2>&1; then
      latexmk -C || true
    fi
    rm -f main.{aux,bbl,blg,fls,fdb_latexmk,log,lof,lot,out,toc,pdf} || true
    ;;
  *)
    echo "Usage: $0 [pdf|clean|clobber]" >&2
    exit 2
    ;;
esac

