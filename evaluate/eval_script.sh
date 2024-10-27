#!/bin/bash
#source ../.venv/bin/activate


# Function for generating sampling model names
models=()
function generate_sampling_names() {
  base_models=$@
#  methodologies=( "forest_fire_v2" "random_walk_v2" "random_jump_v2" "time_v2" "forest_fire_binomial_v2" "pinsage_v2" )
  methodologies=( "forest_fire_binomial_v2" "forest_fire_v2" "random_walk_v2" )
  args="1 2 3 4"
#  ratios="0.05 0.1 0.2 0.5"
  ratios="0.1 0.2"
  ml="" # model list
  for bm in $base_models; do
    for m in "${methodologies[@]}"; do
      for r in $ratios; do
        if [[ -n $args ]]; then
          for a in $args; do
            nm="${bm}-${m}-${a}-${r}"
            ml="${ml[@]} $nm"
          done
        fi
        # add base model with default seed too.
#        nm="${bm}-${m}-${r}"
#        ml="${ml[@]} $nm"
      done
    done
  done
  models=$ml
}

rm -f eval_logs.txt

RESULT_PATH="../sampling_results"
FEATURE_CONF="transsage"

settings="standard"

## ------- METHOD USING FEATURES
bm=("ginrec" "pinsage")
generate_sampling_names "${bm[@]}"

datasets=( "ml_1m_temporal" "ab_temporal" "yk_temporal" )
parameters=( "" "ml_1m_temporal" "ml_1m_temporal" )

max_processes=4
if false; then
  for i in "${!datasets[@]}"; do
    set -e
    dataset=${datasets[i]}
    parameter=${parameters[i]}
    ext="features_${FEATURE_CONF}${parameter:+_parameter_${parameter}}"

    set +e
    for setting in $settings; do
      if [[ -n $other_models ]]; then
        for model in $models; do
          for other_model in $other_models; do
            if [[ $model == $other_model* ]]; then
              cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${model} --folds fold_0 --experiment_setting ${setting} --feature_configuration ${FEATURE_CONF} ${parameter:+--parameter ${parameter}} ${other_model:+--other_model ${other_model}}"
              cmd="$cmd &>> eval_logs.txt"
              echo $cmd
            fi
          done
        done
      else
          cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${models} --folds fold_0 --experiment_setting ${setting} --feature_configuration ${FEATURE_CONF} ${parameter:+--parameter ${parameter}}"
          echo $cmd
      fi
    done
  done | xargs --max-args=1 --max-procs=$max_processes -i -t bash -c "{}"
fi

models=()
bm=( "inmo-uni" )
generate_sampling_names "${bm[@]}"

for i in "${!datasets[@]}"; do
  dataset=${datasets[i]}
  parameter=${parameters[i]}
  ext="${parameter:+parameter_${parameter}}"

  set +e
  for setting in $settings; do
    if [[ -n $other_models ]]; then
      for model in $models; do
        for om in $other_models; do
          if [[ $model == $om* ]]; then
            other_model=$om
            cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${model} --folds fold_0 --experiment_setting ${setting} ${parameter:+--parameter ${parameter}} ${other_model:+--other_model ${other_model}}"
            cmd="$cmd &>> eval_logs.txt"
            echo $cmd;
          fi
        done
      done
    else
        cmd="python3 metric_calculator.py --data ../datasets --results_path ${RESULT_PATH} --experiments ${dataset} --include ${models} --folds fold_0 --experiment_setting ${setting} ${parameter:+--parameter ${parameter}}"
        echo $cmd;
    fi
  done | xargs --max-args=1 --max-procs=$max_processes -i -t bash -c "{}"
done
wait
