#!/bin/bash

set -x

#model_sizes=("MP-S" "OFF-S")
num_features=(32 64 96 128 160 192)

export IMPLEMENTATION=e3j 

out_file="${IMPLEMENTATION}.out"
echo "" > $out_file
for model_size in "${num_features[@]}"; do
  NUM_FEATURES=$model_size python bench_cuex_mario.py >> $out_file
done



export IMPLEMENTATION=cuex 

out_file="${IMPLEMENTATION}.out"
echo "" > $out_file
for model_size in "${num_features[@]}"; do
  NUM_FEATURES=$model_size python bench_cuex_mario.py >> $out_file
done