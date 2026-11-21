#!/usr/bin/env bash
# Pin the arena to one GPU on a shared machine, then run the whole roster.
#
#   ARENA_GPU_UUID=GPU-xxxxxxxx-.... ./run_gpu.sh
#   ARENA_GPU_UUID=GPU-xxxxxxxx-.... ONLY=hf REPEATS=5 TEMP=0.7 ./run_gpu.sh
#
# CUDA_VISIBLE_DEVICES must be a GPU *UUID*, not an index. On a mixed-card machine
# the CUDA index ordering does not match nvidia-smi s, and CUDA_DEVICE_ORDER=PCI_BUS_ID
# did not fix it either: index 2 still selected a different card, which scattered a
# model onto a Turing card with no FlashAttention and then CPU-offloaded a 32B.
# Find yours with:  nvidia-smi --query-gpu=index,name,uuid --format=csv
set -u
: "${ARENA_GPU_UUID:?set ARENA_GPU_UUID to a GPU UUID from nvidia-smi --query-gpu=uuid --format=csv}"
export CUDA_VISIBLE_DEVICES="$ARENA_GPU_UUID"
export ONLY="${ONLY:-all}" REPEATS="${REPEATS:-5}" TEMP="${TEMP:-0.7}"
cd "$(dirname "$0")"
exec ./run_all.sh
