from copy import deepcopy

from models.inmo.inmo_recommender import INMORecommender
from models.ginrec.ginrec_recommender import GInRecRecommender
from models.pinsage.pinsage_recommender import PinSAGERecommender

inductive_methods = {
    'ginrec': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'n_layer_samples': [25, 10, 5],
        'hyperparameters':  {
            'learning_rates': [0.1, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [1e-8, 2.],
            'gate_types': ['concat'],
            'weight_decays': [1e-12, 1.],
            'aggregators': ['bi-interaction'],
            'use_ntype': [False],
            'sampling_on_inference': [False],
            'l2_loss': [0],
            'disentanglement': [False],
            # 'disentanglement_weight': [0.1, 0.01, 0.001, 1e-12, 0.2, 0.5, 1, 0],
            'sample_time': [False],
            'timed_batching': [True, False],
            'use_global': ['mean', 'none'],
            'normalizations': ['none'],
            'neighbor_sampling_methods': ['none'],
            'local_names': ['mean', 'none'],
            'neg_sampling_methods': ['uniform'],
            'layers': [4, 3, 2],
            'optimizers': ['AdamW'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice', 'l2_loss': 'choice',
                         'disentanglement': 'choice', 'disentanglement_weight': 'choice',
                         'sampling_on_inference': 'choice', 'use_ntype': 'choice',
                         'sample_time': 'choice', 'use_global': 'choice', 'timed_batching': 'choice',
                         'use_local': 'choice', 'normalizations': 'choice', 'neighbor_sampling_methods': 'choice',
                         'local_names': 'choice', 'neg_sampling_methods': 'choice', 'layers': 'choice',
                         'optimizers': 'choice'}
    },
    'inmo': {
        'model': INMORecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates':  [1e-4, 0.005],
            'weight_decays': [1e-5, 100.],
            'dropouts': [0., 0.8],
            'n_layers': [1, 2, 3],
            'auxiliary_weight': [1e-4, 1],
            'dim': [64],
            'sampling_method': ['uniform']
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous', 'weight_decays': 'continuous',
                         'attention_samples': 'choice', 'dims': 'choice', 'auxiliary_weight': 'continuous',
                         'dim': 'choice', 'n_layers': 'choice', 'sampling_method': 'choice'}
    },
    'pinsage': {
        'model': PinSAGERecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [1.e-5, 0.1],
            'dropouts': [0., 0.8],
            'weight_decays': [1e-8, .1],
            'deltas': [0.01, 32.]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'deltas': 'continuous'},
        'features': True
    },
}


# definition of sampling base
samplers = ['forest_fire_v2', 'random_walk_v2', 'random_jump_v2', 'time_v2', 'forest_fire_binomial_v2', 'pinsage_v2']
ratios = ['0.05', '0.1', '0.2', '0.5']
seeds = ['1', '2', '3', '4']
sampling_models = {
    'ginrec': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'features': True
    },
    'inmo': {
        'model': INMORecommender,
        'use_cuda': True,
        'graphs': ['cg_pos']
    },
    'pinsage': {
        'model': PinSAGERecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'features': True
    }
}

# Script for generating model combinations of samling models

def _assign_model(mc, m, s, r, sd=None):
    model_config = deepcopy(mc)
    sm = s + ('' if sd is None else f'-{sd}')
    sg = s + ('' if sd is None else f'_{sd}')
    model_name = f'{m}-{sm}-{r}'
    graphs = [f'{g}_{sg}_{r}' for g in model_config['graphs']]
    model_config.update({
        'graphs': graphs,
        'hyperparameters': {},
        'sherpa_types': {}
    })
    model_config['ablation_parameter'] = model_config.get('ablation_parameter', {})
    return {model_name: model_config}


sampling_ablation = {}
for sampler in samplers:
    for ratio in ratios:
        for model, model_config in sampling_models.items():
            sampling_ablation.update(_assign_model(model_config, model, sampler, ratio))  # base graph
            for seed in seeds:
                sampling_ablation.update(_assign_model(model_config, model, sampler, ratio, seed))  # ablation


# Update dictionary with ablation studies
dgl_models = {}
dgl_models.update(inductive_methods)
dgl_models.update(sampling_ablation)