import concurrent
import itertools
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from shared.utility import get_experiment_configuration

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', help='datasets to run',
                    default=['ml_1m_temporal', 'ab_temporal', 'yk_temporal'])
parser.add_argument('--methods', nargs='+', help='methods to run',
                    default=["ginrec", "pinsage", "inmo"])
parser.add_argument('--features', nargs='+', help='features to run', default=['transsage', 'transsage', 'transsage'])
parser.add_argument('--results_path', help='path to store results', default='./results')
parser.add_argument('--folds', nargs='+', help='folds to run', default=['fold_0'])
parser.add_argument('--gpus', nargs='+', help='gpus to run', default=['0', '1', '2', '3'])
parser.add_argument('--eval', action='store_true', help='flag for training')
parser.add_argument('--skip', action='store_true', help='flag for skipping trained/test folds')
parser.add_argument('--state', nargs='+', help='state to run', default=[])
parser.add_argument('--parameter', nargs='+', help='parameter to run', default=['', 'ml_1m_temporal', 'ml_1m_temporal'])
parser.add_argument('--value', nargs='+', help='parameter value to run', default=['', '1', '1'])
parser.add_argument('--other_model', type=json.loads, help='other model to run', default='{}')
parser.add_argument('--model_value', help='other model value to run', default='')
parser.add_argument('--samplers', nargs='+', help='sampling methods to run', default=['forest_fire_v2', 'random_walk_v2', 'random_jump_v2', 'time_v2', 'forest_fire_binomial_v2', 'pinsage_v2'])
parser.add_argument('--ratios', nargs='+', help='other model usage to run', default=['0.05', '0.1', '0.2', '0.5'])
parser.add_argument('--apply_sampling', action='store_true', help='flag for applying sampling')
parser.add_argument('--test_workers', default=2, type=int, help='number of workers for testing')
parser.add_argument('--graph_seeds', nargs='+', help='graph seeds to run', default=['0'])

args = parser.parse_args()
g_datasets = args.datasets
g_methods = args.methods
dataset_feature_list = args.features
RESULTS = args.results_path
g_folds = args.folds
g_gpus = args.gpus
TRAINING = not args.eval
STATE_LIST = args.state
PARAMETER_LIST = args.parameter
PARAMETER_VALUES = args.value
OTHER_MODEL = args.other_model
MODEL_VALUE = args.model_value
SKIP_TRAINED_FOLDS = args.skip
TEST_WORKERS = args.test_workers
GRAPH_SEEDS = args.graph_seeds

DATASET_PARAMS = {'feats': dataset_feature_list, 'params': PARAMETER_LIST, 'states': STATE_LIST,
                  'values': PARAMETER_VALUES}

if args.apply_sampling:
    base_methods = g_methods
    samplers = args.samplers
    ratios = args.ratios
    g_methods = []
    OTHER_MODEL = {}
    for r in ratios:
        for s in samplers:
            for m in base_methods:
                g_methods.append(f'{m}-{s}-{r}')
                OTHER_MODEL[g_methods[-1]] = m
                if len(GRAPH_SEEDS) > 1 and GRAPH_SEEDS[0] != '0':
                    for seed in GRAPH_SEEDS:
                        if seed == '0':
                            continue

                        g_methods.append(f'{m}-{s}-{seed}-{r}')
                        OTHER_MODEL[g_methods[-1]] = m


def runner(folds=None, datasets=None, methods=None, gpu=None):
    gpu = gpu if gpu is not None else 0

    features = datasets.get('feats', '')
    state = datasets.get('states', '')
    parameter = datasets.get('params', '')
    value = datasets.get('values', '')
    datasets = datasets['dataset']
    other_model = OTHER_MODEL.get(methods, '')

    experiment = get_experiment_configuration(datasets)
    if TRAINING:
        func = lambda x: x.endswith('state.pickle') and model_name in x
    else:
        func = lambda x: x.endswith('_predictions.pickle') and model_name in x

    methods_require_feature = any([methods.startswith(m) for m in ['ginrec', 'pinsage', 'graphsage']])

    workers = 4 if methods != 'igmc' else 8  # IGMC is very slow and therefore needs more workers
    model_name = '_'.join(filter(lambda x: bool(x), [
        methods,
        f'features_{features}' if features and methods_require_feature  else '',
        state,
        f'parameter_{parameter}' if parameter else '',
        f'model_{other_model}' if other_model else ''
    ]))
    p = os.path.join(RESULTS, experiment.name, methods)
    if SKIP_TRAINED_FOLDS and os.path.isdir(p) and len(list(filter(func, os.listdir(p)))):
        print(f'Skipping {p} with gpu {gpu}')
        return gpu

    if TRAINING:
        str_arg = (f"CUDA_VISIBLE_DEVICES={gpu} python3 train/dgl_trainer.py --data ./datasets --out_path {RESULTS} "
                   f"--experiments {experiment.name} --include {methods} --test_batch 1024 --workers={workers} "
                   f"--folds {folds} --feature_configuration {features}")
        str_arg += f" --parameter {parameter}" if parameter else ""
        str_arg += f" --other_model {other_model}" if other_model else ""
    else:

        str_arg = f"CUDA_VISIBLE_DEVICES={gpu} python3 evaluate/dgl_evaluator.py --data ./datasets --results_path {RESULTS} "\
                    f"--experiments {experiment.name} --include {methods} --test_batch 1024 --workers {workers} "\
                    f"--folds {folds} --feature_configuration {features} --require_state"
        str_arg += f" --parameter {parameter}" if parameter else ""
        str_arg += f" --parameter_usage {value}" if value else ""
        str_arg += f" --other_model {other_model}" if other_model else ""
        str_arg += f" --other_model_usage {MODEL_VALUE}" if MODEL_VALUE else ""
        str_arg += f" --max_processes {TEST_WORKERS}"

    p = subprocess.Popen(str_arg, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in p.stdout:
        print(line)

    p.wait()

    return gpu


def run():
    ngpus = len(g_gpus)

    params = []
    for i in range(len(g_datasets)):
        d_params = {'dataset': g_datasets[i]}
        for key, value in DATASET_PARAMS.items():
            if value:
                d_params[key] = value[i]
        params.append(d_params)

    combinations = list(itertools.product(g_folds, params, g_methods))

    futures = []
    first = True
    with ThreadPoolExecutor(max_workers=ngpus) as e:
        while combinations:
            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for _, gpu in list(zip(combinations, g_gpus)):
                    futures.append(e.submit(runner, *combinations.pop(0), gpu))

                first = False
            else:
                # Check if any completed
                completed = list(filter(lambda x: futures[x].done(), range(len(futures))))

                # if any process is completed start new on same gpu; otherwise, wait for one to finish
                if completed:
                    f = futures.pop(completed[0])
                    gpu = f.result()
                    futures.append(e.submit(runner, *combinations.pop(0), gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)


if __name__ == '__main__':
    run()