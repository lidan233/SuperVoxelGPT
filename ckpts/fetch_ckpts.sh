#!/usr/bin/env bash
# Fetch the released checkpoints into this directory.
#
# Purpose: the weights are 8.2 GiB and live outside git. This pulls them and
#     checks them, so a truncated or half-synced download fails here rather than
#     several minutes into a run with an error about tensor shapes.
#
# Input:  an rclone remote that can see the release folder (default `box:`).
#         The published share link is login-gated, so an anonymous curl/wget
#         cannot be used; see README.md in this directory.
# Output: this directory, populated and verified against MD5SUMS.txt.
#
# Key idea: verification is not optional and not a separate step someone may
#     forget. Every file is checked against the manifest before the script
#     reports success, because the failure a partial download produces later is
#     silent — a checkpoint that loads with missing keys still runs, and only the
#     numbers come out wrong.
#
#   bash fetch_ckpts.sh
#   REMOTE=mybox:some/path bash fetch_ckpts.sh
#   RCLONE_FLAGS="--transfers 4" bash fetch_ckpts.sh
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${REMOTE:-box:text2shape_supervoxel}"
# Few large files: more parallel streams per file beats more files at once.
RCLONE_FLAGS="${RCLONE_FLAGS:---transfers 2 --multi-thread-streams 8 --progress}"

command -v rclone >/dev/null 2>&1 || {
  echo "rclone not found. Install it and configure a remote that can see the" >&2
  echo "release folder: https://rclone.org/box/" >&2; exit 1; }

echo "== fetching from $REMOTE =="
# --exclude README.md: this directory's README ships with the repository, not with
# the weights.
# shellcheck disable=SC2086
rclone copy "$REMOTE" . --exclude README.md $RCLONE_FLAGS

[ -f MD5SUMS.txt ] || { echo "MD5SUMS.txt did not arrive; the remote path is probably wrong" >&2; exit 1; }

echo
echo "== verifying =="
if md5sum -c MD5SUMS.txt; then
  echo
  echo "OK — $(du -sh . | cut -f1) in $(pwd)"
  echo "Point the pipeline at it with --ckpts, or leave it here (the default)."
else
  echo
  echo "VERIFICATION FAILED. Re-run this script: rclone skips files that already" >&2
  echo "match, so only the bad ones transfer again." >&2
  exit 1
fi
