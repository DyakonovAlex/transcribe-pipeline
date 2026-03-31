#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_debug_compare.sh [options] /path/to/meeting.m4a
       run_debug_compare.sh --help

Скрипт для отладки: venv, зависимости и вызов scripts/transcribe_meeting.py
с удобными значениями по умолчанию (temp-dir, cleanup).

Options:
  --temp-dir DIR              Каталог для WAV/транскрипта (по умолчанию: /tmp/transcribe_debug_<stem>)
  --compare-ollama CSV        Список моделей через запятую (передаётся как --compare-ollama-model)
  --compare-whisper CSV       Список путей whisper-cli через запятую (--compare-whisper-cli)
  --start-stage STAGE         convert | transcribe | ollama | update (по умолчанию: convert)
  --end-stage STAGE           (по умолчанию: update)
  --cleanup MODE              always | on-success | never (по умолчанию: never)
  --config PATH               Путь к config.yaml (по умолчанию: config.yaml в корне проекта)
  --                          Всё после -- передаётся в Python как есть

Examples:
  ./run_debug_compare.sh --compare-ollama "model-a,model-b" "/path/to/meeting.m4a"
  ./run_debug_compare.sh --start-stage ollama --end-stage update \\
    --temp-dir /tmp/my-debug --compare-ollama "model-a,model-b" "/path/to/meeting.m4a"
EOF
}

AUDIO=""
TEMP_DIR=""
COMPARE_OLLAMA=""
COMPARE_WHISPER=""
START_STAGE=""
END_STAGE=""
CLEANUP=""
CONFIG=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --temp-dir)
      TEMP_DIR="${2:?missing value for --temp-dir}"
      shift 2
      ;;
    --compare-ollama)
      COMPARE_OLLAMA="${2:?missing value for --compare-ollama}"
      shift 2
      ;;
    --compare-whisper)
      COMPARE_WHISPER="${2:?missing value for --compare-whisper}"
      shift 2
      ;;
    --start-stage)
      START_STAGE="${2:?missing value for --start-stage}"
      shift 2
      ;;
    --end-stage)
      END_STAGE="${2:?missing value for --end-stage}"
      shift 2
      ;;
    --cleanup)
      CLEANUP="${2:?missing value for --cleanup}"
      shift 2
      ;;
    --config)
      CONFIG="${2:?missing value for --config}"
      shift 2
      ;;
    --)
      shift
      EXTRA+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -n "$AUDIO" ]]; then
        echo "Error: unexpected extra argument: $1" >&2
        exit 2
      fi
      AUDIO="$1"
      shift
      ;;
  esac
done

if [[ -z "$AUDIO" ]]; then
  echo "Error: path to audio file is required." >&2
  usage
  exit 2
fi

if [[ ! -f "$AUDIO" ]]; then
  echo "Error: input file not found: $AUDIO" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "requirements.txt" || ! -f "config.yaml" || ! -f "scripts/transcribe_meeting.py" ]]; then
  echo "Error: run this script from the project root (missing expected files)." >&2
  exit 2
fi

STEM="$(basename "$AUDIO")"
STEM="${STEM%.*}"
if [[ -z "$STEM" ]]; then
  STEM="meeting"
fi

if [[ -z "$TEMP_DIR" ]]; then
  TEMP_DIR="/tmp/transcribe_debug_${STEM}"
fi

if [[ -z "$CLEANUP" ]]; then
  CLEANUP="never"
fi

if [[ -z "$START_STAGE" ]]; then
  START_STAGE="convert"
fi

if [[ -z "$END_STAGE" ]]; then
  END_STAGE="update"
fi

# Ensure venv exists
if [[ ! -x "venv/bin/python" ]]; then
  echo "Creating virtual environment in ./venv ..." >&2
  python3 -m venv venv
fi

echo "Installing Python dependencies ..." >&2
./venv/bin/python -m pip install -r requirements.txt >/dev/null

PY_ARGS=(
  "$AUDIO"
  "--temp-dir" "$TEMP_DIR"
  "--start-stage" "$START_STAGE"
  "--end-stage" "$END_STAGE"
  "--cleanup" "$CLEANUP"
)

if [[ -n "$CONFIG" ]]; then
  PY_ARGS+=("--config" "$CONFIG")
fi
if [[ -n "$COMPARE_OLLAMA" ]]; then
  PY_ARGS+=("--compare-ollama-model" "$COMPARE_OLLAMA")
fi
if [[ -n "$COMPARE_WHISPER" ]]; then
  PY_ARGS+=("--compare-whisper-cli" "$COMPARE_WHISPER")
fi

echo "Debug temp directory: $TEMP_DIR" >&2
echo "Running: transcribe_meeting.py ${PY_ARGS[*]} ${EXTRA[*]:-}" >&2

exec ./venv/bin/python scripts/transcribe_meeting.py "${PY_ARGS[@]}" "${EXTRA[@]}"
