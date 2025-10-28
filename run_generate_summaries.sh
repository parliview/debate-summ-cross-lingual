#!/bin/bash

python experiments/generate_debate_summaries.py \
  --input-dir data/debates \
  --model-type 'ollama' \
  --model-name 'phi4:14b' \
  --src-model 'phi4:14b' \
  --zero-shot 

python experiments/generate_debate_summaries.py \
  --input-dir data/debates \
  --model-type 'ollama' \
  --model-name 'phi4:14b' \
  --src-model 'phi4:14b' \
  --structured 

python experiments/generate_debate_summaries.py \
  --input-dir data/debates \
  --model-type 'ollama' \
  --model-name 'phi4:14b' \
  --src-model 'phi4:14b' \
  --structured \
  --hierarchical \


python experiments/reconstruct_positions.py \
  --model-name 'qwen3:30b-a3b' \
  --model-type 'ollama' \
  --input-dir data/debates

