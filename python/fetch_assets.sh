#!/bin/sh
# Download the pinned checkpoint and held-out text, then verify every file against assets.sha256.
set -e
cd "$(dirname "$0")/.."
REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
DREV=b08601e04326c79dfdd32d625aee71d232d685c3
mkdir -p model data
for f in README.md config.json generation_config.json merges.txt model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  [ -f "model/$f" ] || curl -fL -o "model/$f" "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/$REV/$f"
done
[ -f data/wikitext2_test.parquet ] || curl -fL -o data/wikitext2_test.parquet \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/$DREV/wikitext-2-raw-v1/test-00000-of-00001.parquet"
sha256sum -c assets.sha256
