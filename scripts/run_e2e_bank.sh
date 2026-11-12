#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT or copy .env.example to .env}"
: "${GOOGLE_CLOUD_LOCATION:?Set GOOGLE_CLOUD_LOCATION or copy .env.example to .env}"
: "${VERTEX_MODEL:?Set VERTEX_MODEL or copy .env.example to .env}"

# Use the OpenMP library already bundled with scikit-learn on this macOS setup.
sklearn_lib="$repo_root/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
if [[ -d "$sklearn_lib" ]]; then
  export DYLD_LIBRARY_PATH="$sklearn_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

export LANGSMITH_TRACING=false
export AGENTIC_GAN_PROGRESS=1
exec .venv/bin/python -u -m agentic_gan.orchestrator --dataset bank_marketing
