#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=../../tool

output_root=../out

mkdir -p ${output_root}

speakers=(rosedoodle)

for speaker in "${speakers[@]}"; do
  output_dir=${output_root}/${speaker}
  mkdir -p "${output_dir}"
  python -m clip \
    --transcription_dir="../../02_human_filtering/out/${speaker}" \
    --audio_dir="../../data/wav_22.05k/${speaker}" \
    --output_dir="${output_dir}"
done
