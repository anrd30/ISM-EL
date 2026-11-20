#!/usr/bin/env bash
# Run every model N times and aggregate. Repeats need temperature > 0, or the
# decoder is greedy and the runs are identical.
set -u
cd "$(dirname "$0")"

REPEATS="${REPEATS:-5}"
TEMP="${TEMP:-0.7}"

OLLAMA=( "qwen2.5-coder:1.5b" "qwen2.5-coder:7b" "llama3.1:8b" )
HF=(
  "Qwen/Qwen2.5-Coder-14B-Instruct"
  "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
  "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
  "Qwen/Qwen2.5-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "microsoft/Phi-3-mini-128k-instruct"
  "google/gemma-3-4b-it"
  "arcee-ai/AFM-4.5B"
  "BioMistral/BioMistral-7B"
)

echo "=== started $(date -Is): ${REPEATS} repeats at temperature ${TEMP} ==="
python3 -m arena.run verify || { echo "VERIFY FAILED"; exit 1; }

run_one() {  # $1 = backend spec, $2 = short name
  local spec="$1" name="$2" k out
  for k in $(seq 1 "$REPEATS"); do
    out="arena/submissions/${name}#${k}"
    echo; echo "--- ${name} repeat ${k}/${REPEATS} ---"
    if python3 -m arena.generate --model "$spec" --out "$out" --temperature "$TEMP" --seed "$k"; then
      python3 -m arena.run eval --submission "${out}/policy.py" --name "${name}#${k}" \n        || echo "eval failed: ${name}#${k} (recorded)"
    else
      echo "no policy extracted: ${name}#${k}"
    fi
  done
}

for m in "${OLLAMA[@]}"; do run_one "ollama:${m}" "$(echo "$m" | tr ":/" "--")"; done
for m in "${HF[@]}"; do
  run_one "hf:${m}" "$(echo "$m" | tr ":/" "--")"
  python3 - <<PY
import gc, torch
gc.collect(); torch.cuda.empty_cache()
PY
done

echo; echo "=== per model ==="; python3 -m arena.run report --by-model
echo; echo "=== every run ===";  python3 -m arena.run report
echo "=== finished $(date -Is) ==="
