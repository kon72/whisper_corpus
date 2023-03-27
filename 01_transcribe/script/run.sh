#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=../../tool

output_root=../out

mkdir -p ${output_root}

speakers=(rosedoodle)

for speaker in "${speakers[@]}"; do
  output_dir=${output_root}/${speaker}
  mkdir -p "${output_dir}"
  python -m transcribe \
    --model=medium \
    --input_dir="../../data/wav_16k/${speaker}" \
    --output_dir="${output_dir}" \
    > >(tee "${output_dir}/transcribe_stdout.log") \
    2> >(tee "${output_dir}/transcribe_stderr.log" >&2)
done
