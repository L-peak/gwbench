# Manuscript

Target: *Computers & Geosciences* (Elsevier, subscription route).

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled manuscript is `main.pdf` (25 pages); `main_round2.pdf` is the
matching round-2 revision. The response letter for the second review is
`response_to_reviewers_round2.pdf`. The benchmark protocol and per-well
manifest are in `../docs/BENCHMARK.md` and `../docs/site_manifest.csv`.

## Revision files

- `response_to_reviewers.md` / `.pdf` - response to the first review round
- `response_to_reviewers_round2.md` / `.pdf` - response to the second review
- `highlights.txt` - journal highlights
- `cover_letter.md` - submission cover letter
- `submission_checklist.md` - pre-submission checks
- `figures/fig1..fig8` - manuscript figures (300 dpi)
- `zenodo_doi.txt` - archived release DOI

## Before submission

- Verify the journal's current guide for authors (highlights, graphical
  abstract, declaration sections).
- Confirm the Zenodo archive and Kaggle notebook links are public.
- Recheck that the PDF text extracts correctly (ligatures are disabled).
- Keep the five-mask missing-data and cross-panel result files with the
  submission package.
