#!/usr/bin/env bash
#
# update_from_zip.sh — sync a new package version (delivered as a zip by the
# assistant) into your local git clone, then commit, tag, and push it so the
# GitHub distribution and a GitHub Release are updated.
#
# Usage (run from anywhere on your Mac):
#
#   bash update_from_zip.sh ~/Downloads/ExoplanetAnalysisTools.zip ~/path/to/exoplanet-analysis
#
#   $1 = path to the ExoplanetAnalysisTools.zip you downloaded from the chat
#   $2 = path to your local git clone of the repo (default: current directory)
#
# What it does:
#   1. Unzips the new version to a temporary folder.
#   2. Mirrors its contents into your clone (adding/updating/removing files),
#      while preserving your .git directory and anything git-ignored.
#   3. Shows you the diff summary and asks for confirmation.
#   4. Commits, creates a version tag from the package version, and pushes.
#
# Requires: git, unzip, rsync (all standard on macOS).

set -euo pipefail

ZIP="${1:?Usage: bash update_from_zip.sh <zip> <repo-dir>}"
REPO="${2:-$(pwd)}"

if [ ! -f "$ZIP" ]; then
  echo "Error: zip not found: $ZIP" >&2; exit 1
fi
if [ ! -d "$REPO/.git" ]; then
  echo "Error: $REPO is not a git repository (no .git). Clone it first:" >&2
  echo "  git clone https://github.com/edermartioli/exoplanet-analysis.git" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Unzipping $ZIP"
unzip -q "$ZIP" -d "$TMP"

# The zip contains a top-level folder (ExoplanetAnalysisTools/). Find it.
SRC="$(find "$TMP" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
if [ -z "$SRC" ]; then
  echo "Error: could not find the package folder inside the zip." >&2; exit 1
fi
echo "==> Package folder: $SRC"

# Mirror into the repo. --delete removes files that no longer exist in the new
# version; the excludes protect your git metadata and local/generated files.
echo "==> Syncing into $REPO"
rsync -a --delete \
  --exclude ".git/" \
  --exclude ".github/workflows/*.local.yml" \
  --exclude "notebooks/outputs/" \
  --exclude "notebooks/data/TOI-3568/" \
  --exclude "notebooks/data/WASP-108/" \
  --exclude "notebooks/data/TOI-1736/" \
  --exclude "**/__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  "$SRC"/ "$REPO"/

cd "$REPO"

# Read the new version from the single source of truth.
VERSION="$(python3 -c "import re; s=open('src/exoplanet_analysis/__init__.py').read(); print(re.search(r'__version__\s*=\s*[\"\\']([^\"\\']+)[\"\\']', s).group(1))")"
echo "==> New package version: $VERSION"

echo
echo "==> Changes to be committed:"
git add -A
git status --short
echo

read -r -p "Commit, tag v$VERSION, and push? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
  echo "Aborted. Your working tree has the staged changes; nothing was pushed."
  exit 0
fi

git commit -m "Release v$VERSION"

# Create the tag only if it does not already exist.
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
  echo "Warning: tag v$VERSION already exists; pushing commit without re-tagging."
  git push
else
  git tag "v$VERSION"
  git push && git push --tags
fi

echo
echo "==> Done. Pushed to GitHub."
echo "    The release workflow will build the package and publish a GitHub Release for v$VERSION."
echo "    Watch it at: https://github.com/edermartioli/exoplanet-analysis/actions"
