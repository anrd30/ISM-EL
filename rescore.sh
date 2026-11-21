#!/usr/bin/env bash
# Re-score every submission already on disk, without calling any model again.
# Use after fixing the harness: generation is the expensive half and it is already done.
set -u
cd "$(dirname "$0")"
shopt -s nullglob
for d in arena/submissions/*/; do
  name="$(basename "$d")"
  case "$name" in __pycache__|_*) continue ;; esac
  [ -f "${d}policy.py" ] || continue
  echo "--- ${name} ---"
  python3 -m arena.run eval --submission "${d}policy.py" --name "${name}" || true
done
echo
echo "=== per model ==="
python3 -m arena.run report --by-model
