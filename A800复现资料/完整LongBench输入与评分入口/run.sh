#!/usr/bin/env bash
set -Eeuo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec python3 "$kit/recover.py" rescue "$@"
