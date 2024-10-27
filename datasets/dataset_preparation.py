import argparse
import os
import subprocess
from typing import List

from configuration.experiments import experiment_names
from shared.configuration_classes import ExperimentConfiguration
from shared.utility import valid_dir, get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='Path to datasets', default='.')
parser.add_argument('--experiments', nargs='+', choices=experiment_names, help='Experiment to create.')


def subprocess_runner(args: str):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        for line in p.stdout:
            print(line.decode('utf-8'))


def amazon_runner(path: str, experiment: ExperimentConfiguration):
    # Run og and kgat downloaders
    out_path = os.path.join(path, 'amazon-book')
    os.makedirs(out_path, exist_ok=True)
    og_args = f'python downloaders/ab-downloader-og.py --out_path {out_path}'
    kgat_args = f'python downloaders/ab-downloader-kgat.py --out_path {out_path}'

    subprocess_runner(og_args)
    subprocess_runner(kgat_args)

    # Run og and kgat converters
    og_args = f'python downloaders/ab_converter_og.py --path {out_path}'
    kgat_args = f'python downloaders/ab_converter_kgat.py --path {out_path}'

    subprocess_runner(og_args)
    subprocess_runner(kgat_args)


def movielens_runner(path, experiment: ExperimentConfiguration):
    # Run ml downloader
    out_path = os.path.join(path, 'movielens')
    os.makedirs(out_path, exist_ok=True)
    ml_args = f'python downloaders/ml-downloader.py --out_path {out_path}'

    subprocess_runner(ml_args)

    # Run ml converter
    ml_args = f'python downloaders/ml_converter.py --path {out_path}'

    subprocess_runner(ml_args)


def preprocessor_runner(path: str, experiment: ExperimentConfiguration):
    # Run preprocessor
    arg = f'python preprocessors/preprocessor.py --path {path} --dataset {experiment.dataset.name}'
    subprocess_runner(arg)


def partitioner_runner(path: str, experiment: ExperimentConfiguration):
    # Run partitioner
    arg = f'python partitioners/partitioner.py --path {path} --experiment {experiment.name}'
    subprocess_runner(arg)


def experiment_to_dgl_runner(path: str, experiment: ExperimentConfiguration):
    # Run experiment_to_dgl
    arg = f'python converters/experiment_to_dgl.py --path {path} --experiment {experiment.name} --graphs_only'
    subprocess_runner(arg)


def feature_extractor_runner(path: str, experiment: ExperimentConfiguration):
    # Run feature_extractor
    dataset_flag = experiment.dataset.name in ['amazon-books', 'yelpkg']
    if dataset_flag:
        features = ['graphsage_scale', 'complex']
    else:
        features = ['graphsage_scale']

    for feature in features:
        arg = (f'python feature_extractors/feature_extractor.py --path {path} --experiment {experiment.name} '
               f'--feature_configurations {feature}')
        subprocess_runner(arg)

    if dataset_flag:
        arg = (f'python feature_extractors/merge_features.py --path {path} --experiment {experiment.name} '
               f'--feature_configuration {" ".join(features)} comsage')
        subprocess_runner(arg)


def run(path: str, experiments: List[str]):
    for experiment in experiments:
        experiment = get_experiment_configuration(experiment)
        if experiment.dataset.name == 'amazon-book':
            amazon_runner(path, experiment)
        else:
            raise ValueError('Not implemented yet.')

        preprocessor_runner(path, experiment)
        partitioner_runner(path, experiment)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.experiments)