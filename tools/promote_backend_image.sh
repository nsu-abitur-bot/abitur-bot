#!/usr/bin/env bash
# Must run inside the workflow's non-cancelling shared-tag promotion lock.
set -euo pipefail

: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}" "${IMAGE_DIGEST:?}"
# Fail closed on an API error or unexpected response, including manual old reruns.
current_sha=$(gh api "repos/${GITHUB_REPOSITORY}/git/ref/heads/main" --jq '.object.sha')
if [[ "$current_sha" != "$GITHUB_SHA" ]]; then
    echo "Skipping outdated checkout ${GITHUB_SHA}; main is ${current_sha}"
    exit 0
fi

# Use this build's digest, not a SHA tag that another rerun could have rewritten.
docker buildx imagetools create \
    --tag "ghcr.io/${GITHUB_REPOSITORY}:main" \
    "ghcr.io/${GITHUB_REPOSITORY}@${IMAGE_DIGEST}"
