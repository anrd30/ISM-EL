#!/usr/bin/env bash
# Run every model N times and aggregate.
#
# Repeats need temperature > 0: at temperature 0 the decoder is greedy and five
# runs of a model produce five identical files. Each repeat is seeded, so an
# individual run stays reproducible while the set measures how much a model varies.
#
#   REPEATS=5 TEMP=0.7 ./run_all.sh     all models
#   ONLY=ollama ./run_all.sh            skip the HuggingFace models
#   ONLY=hf ./run_all.sh                skip the Ollama models
#
# A model that fails is a result, not a crash: the run continues and the failure
# is recorded in arena/results/.

set -u
cd "$(dirname "$0")"

REPEATS="${REPEATS:-5}"
TEMP="${TEMP:-0.7}"
ONLY="${ONLY:-all}"

OLLAMA_MODELS=(
  "qwen2.5-coder:1.5b"
  "qwen2.5-coder:7b"
  "llama3.1:8b"
)

HF_MODELS=(
  # Only models whose weights are fully present in the local HF cache. A partial
  # download triggers a multi-gigabyte fetch mid-run, and a gated repo fails with a
  # 401 -- neither is a statement about the model, so neither belongs in the table.
  # Verify with: du -sh ~/.cache/huggingface/hub/models--*/
  "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"   # 18G, reasoning, emits <think>
  "BioMistral/BioMistral-7B"                        # 14G, medical finetune, off-domain control
  "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"     # 11G, code-tuned, 4-bit
  "arcee-ai/AFM-4.5B"                               # 8.7G
  "microsoft/Phi-3-mini-128k-instruct"              # 7.2G
  "Qwen/Qwen2.5-3B-Instruct"                        # 5.9G
  # Excluded, incomplete or gated locally:
  #   Qwen/Qwen2.5-Coder-14B-Instruct   779M stub of ~28G
  #   meta-llama/Llama-3.2-3B-Instruct  gated, 401
  #   google/gemma-3-4b-it              3.0G of ~8.6G
  #   unsloth/DeepSeek-R1-Distill-Qwen-32B  5.3G of ~65G
)

short_name() {
  echo "$1" | tr ':/' '--'
}

run_one() {
  local spec="$1"
  local name="$2"
  local k out
  for k in $(seq 1 "$REPEATS"); do
    out="arena/submissions/${name}#${k}"
    echo
    echo "--- ${name} repeat ${k}/${REPEATS} ---"
    if python3 -m arena.generate --model "$spec" --out "$out" --temperature "$TEMP" --seed "$k"; then
      if python3 -m arena.run eval --submission "${out}/policy.py" --name "${name}#${k}"; then
        :
      else
        echo "eval reported a failure for ${name}#${k} (recorded in arena/results/)"
      fi
    else
      echo "no policy extracted for ${name}#${k}; raw reply kept at ${out}/raw_reply.md"
    fi
  done
}

free_gpu() {
  python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
}

echo "=== started $(date -Is): ${REPEATS} repeats at temperature ${TEMP}, models=${ONLY} ==="

if ! python3 -m arena.run verify; then
  echo "VERIFY FAILED -- stopping; results would not be comparable"
  exit 1
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ollama" ]; then
  for m in "${OLLAMA_MODELS[@]}"; do
    run_one "ollama:${m}" "$(short_name "$m")"
  done
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "hf" ]; then
  for m in "${HF_MODELS[@]}"; do
    run_one "hf:${m}" "$(short_name "$m")"
    free_gpu
  done
fi

echo
echo "=== per model ==="
python3 -m arena.run report --by-model
echo
echo "=== every run ==="
python3 -m arena.run report
echo "=== finished $(date -Is) ==="
