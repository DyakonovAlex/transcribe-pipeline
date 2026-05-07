#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [--no-llm] /path/to/meeting.m4a" >&2
}

NO_LLM=0
INPUT_AUDIO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-llm)
      NO_LLM=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -n "$INPUT_AUDIO" ]]; then
        echo "Error: only one input audio file is supported." >&2
        usage
        exit 2
      fi
      INPUT_AUDIO="$1"
      shift
      ;;
  esac
done

if [[ -z "$INPUT_AUDIO" ]]; then
  usage
  exit 2
fi

if [[ ! -f "$INPUT_AUDIO" ]]; then
  echo "Error: input file not found: $INPUT_AUDIO" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "requirements.txt" || ! -f "config.yaml" || ! -f "scripts/transcribe_meeting.py" ]]; then
  echo "Error: run this script from the project root (missing expected files)." >&2
  exit 2
fi

# Ensure venv exists
if [[ ! -x "venv/bin/python" ]]; then
  echo "Creating virtual environment in ./venv ..." >&2
  python3 -m venv venv
fi

# Install dependencies (idempotent)
echo "Installing Python dependencies ..." >&2
./venv/bin/python -m pip install -r requirements.txt >/dev/null

echo "Running transcription pipeline ..." >&2
PYTHON_ARGS=(scripts/transcribe_meeting.py "$INPUT_AUDIO")
if [[ "$NO_LLM" -eq 1 ]]; then
  PYTHON_ARGS+=(--no-llm)
fi
exec ./venv/bin/python "${PYTHON_ARGS[@]}"

