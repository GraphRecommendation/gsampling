#!/bin/bash
DEVICE=1
EXPERIMENTS="ml_1m_stratified ab_stratified yk_stratified"
FEATURES="graphsage complex transr_300 p0_anchor_all_3 comsage transsage"
PPATH="${PWD}/../.."
PYTHON="python3"
for experiment in $EXPERIMENTS; do
  for feature in $FEATURES; do
    echo $feature
    if [[ "$feature" == "complex" || "$feature" == "transr_300" ]] ; then
      cmd="CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=${PPATH} complex_env/bin/python feature_extractor.py --path .. --experiment ${experiment} --feature_configuration ${feature}"
      echo $cmd
      eval $cmd
    elif [[ "$feature" == "comsage"  ||  "$feature" == "transsage" ]] ; then
      if [[ "$feature" == "comsage" ]] ; then
        infeature="complex"
      else
        infeature="transr_300"
      fi
      cmd="CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=${PPATH} ${PYTHON} merge_features.py --path .. --experiment ${experiment} --feature_configurations ${infeature} graphsage ${feature}"
      echo $cmd
      eval $cmd
    else
      cmd="CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=${PPATH} ${PYTHON} feature_extractor.py --path .. --experiment ${experiment} --feature_configuration ${feature}"
      echo $cmd
      eval $cmd
    fi
  done
done
