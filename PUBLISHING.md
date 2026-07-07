# Publishing updates to GitHub

This guide explains how to push a new working version of the package to your
GitHub distribution (<https://github.com/edermartioli/exoplanet-analysis>) from
your MacBook, after we have converged on it here in the chat.

The version number lives in **one place only**:
`src/exoplanet_analysis/__init__.py` (`__version__ = "..."`). `pyproject.toml`
reads it from there automatically, so you never edit the version twice.

Two GitHub Actions workflows are included:

- **CI** (`.github/workflows/ci.yml`) runs on every push to `main`: it installs
  the package, checks it imports, and confirms the command-line tools work.
- **Release** (`.github/workflows/release.yml`) runs when you push a tag like
  `v1.4.2`: it verifies the tag matches the package version, builds the wheel
  and source distribution, and publishes a GitHub Release with those files
  attached and auto-generated notes.

---

## One-time setup on your Mac

You only do this once.

1. Make sure you have the command-line tools: `git`, `python3`, `unzip`,
   `rsync` (all ship with macOS; if `git` prompts you to install the Xcode
   command-line tools, accept).

2. Clone your repository somewhere convenient, e.g.:

   ```bash
   cd ~/Documents
   git clone https://github.com/edermartioli/exoplanet-analysis.git
   cd exoplanet-analysis
   ```

3. Confirm git can push to GitHub (you should have done this already since the
   repo exists). If prompted for credentials, use a Personal Access Token or an
   SSH key — see GitHub's docs on "authenticating with GitHub from Git".

That's it. From now on, updating is a two-minute routine.

---

## The routine: publishing a new version

Each time we finish a new working version here, I give you an updated
`ExoplanetAnalysisTools.zip`. To publish it:

### Option A — the helper script (recommended)

I ship a small script that does everything safely. From your Mac:

```bash
cd ~/Documents/exoplanet-analysis
bash tools/update_from_zip.sh ~/Downloads/ExoplanetAnalysisTools.zip .
```

The script will:

1. unzip the new version,
2. mirror it into your clone (adding, updating, and removing files as needed,
   while leaving your `.git` folder and any generated outputs untouched),
3. show you a summary of what changed and the new version number,
4. ask for confirmation, then commit, tag `vX.Y.Z`, and push.

Pushing the tag triggers the Release workflow, which builds the package and
creates the GitHub Release automatically. You can watch it at
<https://github.com/edermartioli/exoplanet-analysis/actions>.

### Option B — doing it by hand

If you prefer to run the git steps yourself:

```bash
cd ~/Documents/exoplanet-analysis

# 1. Replace the working tree with the new version. Unzip to a temp folder
#    and copy its contents in (the zip has a top-level ExoplanetAnalysisTools/).
unzip -q ~/Downloads/ExoplanetAnalysisTools.zip -d /tmp/eat_new
rsync -a --delete --exclude ".git/" --exclude "notebooks/outputs/" \
      /tmp/eat_new/ExoplanetAnalysisTools/ .

# 2. Check the new version number.
python3 -c "import re; print(re.search(r'__version__\s*=\s*\"([^\"]+)\"', open('src/exoplanet_analysis/__init__.py').read()).group(1))"

# 3. Review, commit, tag, and push (replace X.Y.Z with the version above).
git add -A
git status
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push && git push --tags
```

---

## Bumping the version number

Usually I set the new version in the zip before handing it to you. If you ever
need to bump it yourself:

```bash
python3 tools/bump_version.py patch   # 1.4.1 -> 1.4.2  (bug fixes)
python3 tools/bump_version.py minor   # 1.4.1 -> 1.5.0  (new features)
python3 tools/bump_version.py major   # 1.4.1 -> 2.0.0  (breaking changes)
python3 tools/bump_version.py 1.6.0   # or set it explicitly
```

The script prints the exact git commands to finish the release. Keep
`CHANGELOG.md` updated with a short note for each version — the GitHub Release
notes are generated automatically from your commits, but the changelog is the
human-friendly summary.

---

## What if the tag and version disagree?

The Release workflow deliberately fails if the tag (`v1.4.2`) does not match
`__version__` in the package. This protects you from publishing a mislabeled
release. If it fails: fix `__version__` (or delete and recreate the tag), then
push again.

To move a tag you already pushed:

```bash
git tag -d v1.4.2               # delete locally
git push origin :refs/tags/v1.4.2   # delete on GitHub
git tag v1.4.2                  # recreate on the right commit
git push --tags
```

---

## Notes on the notebooks and data

- The tutorial notebooks are committed **already executed** (with their figures),
  so they render on GitHub.
- The example datasets under `notebooks/data/` are part of the repository
  (about 24 MB total). This is fine for GitHub. If the repo ever grows too large
  for your taste, the natural next step is [Git LFS](https://git-lfs.com) for the
  `.fits` files — ask me and I'll set it up.
- Generated products (`notebooks/outputs/`, `*.h5`, `*_posterior.pars`, pairs
  plots) are git-ignored and never committed.

---

## Optional: publishing to PyPI later

If you eventually want `pip install exoplanet-analysis-tools` to work for anyone
without cloning, we can add a PyPI-publishing step to the Release workflow
(using a PyPI Trusted Publisher, so no passwords are stored). Tell me when you
want that and I'll wire it in.
