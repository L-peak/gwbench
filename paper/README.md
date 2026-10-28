# Manuscript

Target: *Computers & Geosciences* (Elsevier, subscription route).

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled draft is `main.pdf`.

## Before submission

- Replace `[Author One]`, `[Affiliation]`, `[email]`.
- Replace `[repository URL]`, `[Zenodo DOI]`, `[Kaggle URL]`.
- Fill the acknowledgements.
- Verify every citation in `refs.bib` against the published record.
- Check the journal's current guide for authors (highlights, graphical
  abstract, declaration sections).
- Consider adding the cross-site / leave-well-out experiment as a
  supplementary section if time allows.
