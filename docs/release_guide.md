# GitHub, Zenodo, and Kaggle release guide

This guide turns the local project into a citable release. The order matters:
GitHub repository -> GitHub release -> Zenodo DOI -> fill the paper -> make
Kaggle artifacts public if you want to cite them.

## 1. Create the local git repository

From `D:\School\D2` (Git Bash):

```bash
cd /d/School/D2
git init
git add .
git status
```

Check `git status` before committing. The `.gitignore` excludes:

- `data/raw/`, `data/interim/`, `data/processed/` (large data)
- `dist/` (release archives)
- LaTeX build files and `paper/main.pdf`
- Python caches

The paper figures (`paper/figures/*.png`) and source are included.

```bash
git commit -m "gwbench v0.1.0: benchmark code, panels, and manuscript"
git branch -M main
```

## 2. Create the GitHub repository

### Option A: browser

1. Go to <https://github.com/new>.
2. Repository name: `gwbench`.
3. Visibility: Public.
4. Do **not** initialize with README, .gitignore, or license (the local repo
   already has them).
5. Create the repository, then:

```bash
git remote add origin https://github.com/<your-github-username>/gwbench.git
git push -u origin main
```

### Option B: GitHub CLI

```bash
gh auth login          # GitHub.com -> HTTPS -> browser
gh repo create gwbench --public --source=. --remote=origin --push
```

After pushing, copy the repository URL. It replaces `\codeurl` in the paper.

## 3. Create a GitHub release

```bash
git tag -a v0.1.0 -m "gwbench v0.1.0"
git push origin v0.1.0
```

Then on GitHub: **Releases -> Draft a new release -> choose tag v0.1.0 ->
title "gwbench v0.1.0" -> Publish release**.

## 4. Obtain a Zenodo DOI

### Recommended: GitHub integration

1. Sign in to <https://zenodo.org> (GitHub or ORCID login).
2. Open **Settings -> GitHub** and link your GitHub account.
3. Find `gwbench` in the repository list and toggle it **On**.
4. Create the GitHub release from step 3. Zenodo will archive the release and
   mint a DOI automatically (usually within a few minutes).
5. Copy the DOI, e.g. `10.5281/zenodo.1234567`.

### Alternative: manual upload

1. Sign in to Zenodo and click **New upload**.
2. Upload `dist/gwbench-v0.1.0.zip` (6.7 MB).
3. Fill the metadata from `.zenodo.json` (title, creators, description,
   license, keywords).
4. Resource type: Software.
5. Publish. Copy the DOI.

## 5. Fill the placeholders

In `paper/main.tex`:

```latex
\newcommand{\codeurl}{https://github.com/<username>/gwbench}
\newcommand{\zenododoi}{https://doi.org/<DOI>}
\newcommand{\kaggleurl}{https://www.kaggle.com/code/kkksssaaalzy/gwbench-gru-h30}
```

In `CITATION.cff`:

```yaml
repository-code: "https://github.com/<username>/gwbench"
doi: "10.5281/zenodo.<ID>"
```

In `.zenodo.json`, update the `related_identifiers` URL.

Then rebuild:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 6. Kaggle links (optional)

Current private artifacts:

- Dataset: <https://www.kaggle.com/datasets/kkksssaaalzy/gwbench-data>
- 30-day notebook: <https://www.kaggle.com/code/kkksssaaalzy/gwbench-gru-h30>
- 90-day notebook: <https://www.kaggle.com/code/kkksssaaalzy/gwbench-gru-h90>
- Seed study: <https://www.kaggle.com/code/kkksssaaalzy/gwbench-seeds>

Private links are not usable by readers. Either:

- make the dataset and notebooks public from their Kaggle settings pages, or
- remove the Kaggle sentence from the paper and rely on GitHub + Zenodo.

The dataset contains the code and processed USGS-derived panels; raw USGS data
are public. Check the dataset license before making it public.

## 7. Final checklist

- [ ] GitHub repository is public and `README.md` renders.
- [ ] GitHub release `v0.1.0` exists.
- [ ] Zenodo DOI resolves and the archive contains the release zip contents.
- [ ] `paper/main.tex` placeholders replaced; PDF rebuilt.
- [ ] `CITATION.cff` validates (`cffconvert` or GitHub's citation preview).
- [ ] Kaggle artifacts are public or removed from the paper.
- [ ] The Kaggle API token used during development is revoked/regenerated.
