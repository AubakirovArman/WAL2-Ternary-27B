#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# -eq 1 && $1 =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo "Использование: bash $0 owner/new-repository" >&2; exit 2; }
command -v gh >/dev/null || { echo 'Нужен GitHub CLI: gh, затем gh auth login.' >&2; exit 1; }
gh auth status
[[ -d "$repo_root/.git" ]] || { echo 'Сначала создайте локальный Git commit.' >&2; exit 1; }
[[ -z $(git -C "$repo_root" status --porcelain) ]] || { echo 'Сначала сохраните локальные изменения commit.' >&2; exit 1; }
python3 "$repo_root/scripts/verify_evidence.py"
# gh errors if the repository already exists; no existing repo is overwritten.
gh repo create "$1" --public --description 'Ternary 27B QAT research: conversion, native PQ2 inference and reproducible benchmark evidence' \
  --source "$repo_root" --remote origin --push
echo "Опубликовано: https://github.com/$1. Веса и статья на Astana Hub не загружались этим скриптом."
