#!/usr/bin/env bash
# Run a heavy command only while holding a machine-wide lock directory, so model
# inference never overlaps another heavy job (e.g. a film render) on the same Mac.
#   OPENDUBSTAGE_LOCK=/path/to/.heavy.lock scripts/with_heavy_lock.sh <cmd...>
# With OPENDUBSTAGE_LOCK (or the legacy STILLFOLEY_LOCK) unset the command just runs.
set -euo pipefail
lock="${OPENDUBSTAGE_LOCK:-${STILLFOLEY_LOCK:-}}"
if [[ -z "$lock" ]]; then exec "$@"; fi
until mkdir "$lock" 2>/dev/null; do sleep 5; done
echo $$ > "$lock/pid"
trap 'rm -rf "$lock"' EXIT INT TERM
"$@"
