#!/bin/bash
# Apply clang-format to lines changed between two commits.
# Usage: clang-format-diff.sh <head_sha> <base_sha>
#
# Applies formatting in-place to files under include/ (excluding nanorange.hpp)
# using only the diff lines between the merge base and head.

set -euo pipefail

HEAD_SHA="$1"
BASE_SHA="$2"

MERGE_BASE=$(git merge-base "$HEAD_SHA" "$BASE_SHA")
FILES=$(git diff --diff-filter=d --name-only "$MERGE_BASE" | grep ^include | grep -v 'nanorange\.hpp$' || true)
CLANG_FORMAT_DIFF_PATH=$(which clang-format-diff)
echo $FILES | xargs -n1 -t -r git diff -U0 --no-color --relative "$MERGE_BASE" | python3 "$CLANG_FORMAT_DIFF_PATH" -i -p1 -style file
