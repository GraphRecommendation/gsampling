import argparse
import os

import dgl
import torch
from loguru import logger

from datasets.down_samplers import exploration, edge, node

from shared.utility import get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='..')
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--methodology', type=str, required=True)
parser.add_argument('--ratios', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.5])
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--tmp_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)

# possible graphs ckg, cg, kg, cg_pos, ckg_pos
POSTFIXES = ["", "_pos", "_rev", "_pos_rev"]
# RATIOS = [0.05, 0.1, 0.2, 0.5]
RATIOS = [0.5]

def sample_n_save(g, nodes, path, in_path, org_name, methodology):
    eids = g.subgraph(nodes, store_ids=True).edata[dgl.EID]
    out_g = g.edge_subgraph(eids, relabel_nodes=False, store_ids=True)
    dgl.save_graphs(os.path.join(path, f'train_{org_name}_{methodology}.dgl'), [out_g])

    for phase in ['train']: #, 'validation', 'test']:
        if phase != 'train':
            (pg, ), _ = dgl.load_graphs(os.path.join(in_path, f'{phase}_{org_name}.dgl'))
            # dgl.save_graphs(os.path.join(path, f'{phase}_{org_name}_{methodology}.dgl'), [pg])

        info = dgl.data.utils.load_info(os.path.join(in_path, f'{phase}_{org_name}.dgl.info'))
        dgl.data.utils.save_info(os.path.join(path, f'{phase}_{org_name}_{methodology}.dgl.info'), info)


def run_methodology(path, out_path, methodology, cg_nodes, kg_nodes):
    if os.path.isfile(os.path.join(out_path, f'train_cg_pos_{methodology}.dgl')) and \
            os.path.isfile(os.path.join(out_path, f'train_kg_{methodology}.dgl')):
        logger.info(f'{methodology} already run, skipping')
        (cg,), _ = dgl.load_graphs(os.path.join(out_path, f'train_cg_pos_{methodology}.dgl'))
        (kg,), _ = dgl.load_graphs(os.path.join(out_path, f'train_kg_{methodology}.dgl'))
        cg_nodes = torch.unique(torch.cat(cg.edges()))
        kg_nodes = torch.unique(torch.cat(kg.edges()))
        return cg_nodes, kg_nodes
    # Get initial samples
    (cg,), _ = dgl.load_graphs(os.path.join(path, f'train_cg_pos.dgl'))

    # Use sampled nodes to select starting nodes for the knowledge graph.
    # I.e., we want there to be a significant overlap between the sampled cg items and the sampled kg items.
    cg_eids = methodology.initial_sample(cg, cg_nodes)
    methodology.reset()
    src, dst = cg.find_edges(cg_eids)
    cg_nodes = torch.unique(torch.cat([src, dst]))
    (cg_rev, ), _ = dgl.load_graphs(os.path.join(path, f'train_cg_pos_rev.dgl'))
    sample_n_save(cg, cg_nodes, out_path, path, f'cg_pos', f'{methodology}')
    sample_n_save(cg_rev, cg_nodes, out_path, path, f'cg_pos_rev', f'{methodology}')

    (kg,), _ = dgl.load_graphs(os.path.join(path, f'train_kg.dgl'))
    sampled_nodes = torch.zeros_like(cg.nodes(), dtype=torch.bool)
    sampled_nodes[cg_nodes] = True
    nodes = methodology.get_available_nodes(kg).to(torch.bool)
    input_nodes = sampled_nodes & nodes
    kg_eids = methodology.post_sampling(kg, input_nodes.to(torch.float64), kg_nodes)
    methodology.reset()
    src, dst = kg.find_edges(kg_eids)
    kg_nodes = torch.unique(torch.cat([src, dst]))
    sample_n_save(kg, kg_nodes, out_path, path, f'kg', f'{methodology}')

    ckg_nodes = torch.unique(torch.cat([cg_nodes, kg_nodes]))

    for postfix in POSTFIXES:
        (ckg,), _ = dgl.load_graphs(os.path.join(path, f'train_ckg{postfix}.dgl'))
        sample_n_save(ckg, ckg_nodes, out_path, path, f'ckg{postfix}', f'{methodology}')

    return cg_nodes, kg_nodes


def run(path, experiment, methodology, verbose, tmp_dir=None, seed=0):
    experiment = get_experiment_configuration(experiment)
    e_path = os.path.join(path, experiment.dataset.name, experiment.name)

    for f in range(experiment.folds):
        f_path = os.path.join(e_path, f'fold_{f}')
        o_path = f_path if tmp_dir is None else os.path.join(f_path, tmp_dir)

        if not os.path.exists(o_path):
            os.makedirs(o_path)

        cg_nodes, kg_nodes = None, None

        str_arg = 'v2' if seed == 0 else f'v2_{seed}'

        for r in RATIOS:
            if methodology == 'forest_fire':
                m = exploration.forest_fire.ForestFire(r, verbose=verbose, str_arg=str_arg)
            elif methodology == 'forest_fire_binomial':
                m = exploration.forest_fire.ForestFire(r, use_binomial_mean=True, verbose=verbose, str_arg=str_arg)
            elif methodology == 'random_walk':
                m = exploration.random_walk.RandomWalk(r, verbose=verbose, str_arg=str_arg)
            elif methodology == 'random_jump':
                m = exploration.random_walk.RandomWalk(r, use_jumps=True, verbose=verbose, str_arg=str_arg)
            elif methodology == 'time':
                m = edge.time.TimeSampler(r, verbose=verbose, str_arg=str_arg)
            elif methodology == 'pinsage':
                m = node.pinsage_sampler.PinSAGESampler(r, verbose=verbose, str_arg=str_arg)
            else:
                raise ValueError(f'Methodology {methodology} not supported')
            if verbose:
                logger.info(f'Running {m} for fold {f} with ratio {r} on {experiment.name}')

            m.seed(experiment.seed if seed == 0 else seed)
            cg_nodes, kg_nodes = run_methodology(f_path, o_path, m, cg_nodes, kg_nodes)
            logger.info(f'Finished {m} for fold {f} with ratio {r} on {experiment.name}')


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    RATIOS = args.pop('ratios')
    run(**args)