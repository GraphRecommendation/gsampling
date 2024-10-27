import concurrent
import itertools
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

from shared.filelock import FileLock
from shared.utility import get_experiment_configuration

g_datasets = ['ml_1m_temporal']
g_methods = ['ginrec', 'inmo', 'pinsage']
dataset_feature_list = ['transsage']
g_folds = ['fold_0']
g_gpus = ['0']
VENV = False
DRY_RUN = False
TRAINING = True
RESULTS = './results'
RSA_PATH = '~/.ssh/large_rsa'  # change to correct path
WORKERS = 2
STATE_LIST = []
PARAMETER_LIST = []
PARAMETER_VALUES = []
OTHER_MODEL = ''
MODEL_VALUE = ''
SKIP_TRAINED_FOLDS = False
DATASET_PARAMS = {'feats': dataset_feature_list, 'params': PARAMETER_LIST, 'states': STATE_LIST,
                  'values': PARAMETER_VALUES}


def runner(fold=None, dataset=None, method=None, gpu=None):
    gpu = gpu if gpu is not None else 0

    features = dataset.get('feats', '')
    state = dataset.get('states', '')
    parameter = dataset.get('params', '')
    value = dataset.get('values', '')
    dataset = dataset['dataset']

    experiment = get_experiment_configuration(dataset)

    workers = WORKERS if method != 'igmc' else 8  # IGMC is very slow and therefore needs more workers
    str_arg = ""

    if VENV:
        str_arg += "source .venv/bin/activate; "

    str_arg += f"CUDA_VISIBLE_DEVICES={gpu} python3 train/dgl_trainer.py --data ./datasets --out_path {RESULTS} "\
              f"--experiments {experiment.name} --include {method} --test_batch 1024 --workers={workers} "\
              f"--folds {fold} --feature_configuration {features} --parallel"

    if parameter:
        str_arg += f" --parameter {parameter}"
    if OTHER_MODEL:
        str_arg += f" --other_model {OTHER_MODEL}"

    # print(f"Running: {str_arg}")

    p = subprocess.Popen(str_arg, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in p.stdout:
        print(line)

    p.wait()

    return gpu


def has_next(path_to_states):
    try:
        with FileLock(path_to_states + '.lock'):
            if os.path.isfile(path_to_states):
                with open(path_to_states, 'rb') as f:
                    state = pickle.load(f)

                study = state.get('study', None)
                if study is not None:
                    study.next()
        return True
    except StopIteration:
        return False


def method_runner(fold, dataset, method):
    futures = []
    first = True
    ngpus = len(g_gpus)
    os.makedirs(os.path.join(RESULTS, dataset['dataset'], method), exist_ok=True)
    parameter_path = os.path.join(RESULTS, dataset['dataset'], method, 'parameters.states')
    with ThreadPoolExecutor(max_workers=ngpus) as e:
        while has_next(parameter_path):

            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for gpu in g_gpus:
                    futures.append(e.submit(runner, fold, dataset, method, gpu))

                first = False
            else:
                # Check if any completed
                completed = list(filter(lambda x: futures[x].done(), range(len(futures))))

                # if any process is completed start new on same gpu; otherwise, wait for one to finish
                if completed and has_next(parameter_path):
                    f = futures.pop(completed[0])
                    gpu = f.result()
                    futures.append(e.submit(runner, fold, dataset, method, gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)


def run():
    params = []
    for i in range(len(g_datasets)):
        d_params = {'dataset': g_datasets[i]}
        for key, value in DATASET_PARAMS.items():
            if value:
                d_params[key] = value[i]
        params.append(d_params)

    combinations = list(itertools.product(g_folds, params, g_methods))
    for combination in combinations:
        method_runner(*combination)





if __name__ == '__main__':
    run()