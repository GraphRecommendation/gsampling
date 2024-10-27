#!/bin/bash

datasets=( "ml_1m_stratified" "ab_stratified" "yk_stratified" )
methods=( "forest_fire" "random_walk" "random_jump" "time" "forest_fire_binomial" "pinsage" )
seeds="0 1 2 3 4"

time_ratios=1

if (( time_ratios != 0 )); then
  ratios="0.5"
  log_name="gs_logs_ps_ffb.txt"
  tmp_dir="new_time_test"
  rm -f $log_name
  max_processes=1
else
  max_processes=6
fi

for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    if  (( time_ratios == 0 )); then
      for seed in $seeds; do
        cmd="python3 graph_sampler.py --path .. --experiment $dataset --methodology $method --seed $seed"
        echo $cmd
      done
    else
      # Use for timing different ratios
      for ratio in $ratios; do
        cmd="python3 graph_sampler.py --path .. --experiment $dataset --methodology $method --ratios $ratio --verbose --tmp_dir $tmp_dir"
        cmd="/usr/bin/time -v $cmd &>> $log_name"
        echo $cmd
      done
    fi
  done
done | xargs --max-args=1 --max-procs=$max_processes -i -t bash -c "{}"

