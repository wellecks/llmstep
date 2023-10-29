MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES="0.0"
TIMEOUT=600
NUM_SHARDS=1

DATA="data/minif2f.jsonl"
DATASET="minif2f-test"
OUTPUT_DIR="output/minif2f_test"

MODEL="wellecks/llmstep-mathlib4-pythia2.8b"
NAME="wellecks_llmstep-mathlib4-pythia2.8b"

SHARD=0
CUDA_VISIBLE_DEVICES=${SHARD} python proofsearch.py --dataset-name ${DATASET} --temperatures ${TEMPERATURES} --timeout ${TIMEOUT} --num-shards ${NUM_SHARDS} --shard ${SHARD} --model-name ${MODEL} --max-iters ${MAX_ITERS} --dataset-path ${DATA} --num-samples ${NUM_SAMPLES} --early-stop --output-dir ${OUTPUT_DIR} &> ${NAME}_shard${SHARD}.out &
