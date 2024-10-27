import dgl
import torch
from tqdm import tqdm

from datasets.down_samplers.exploration.random_walk import RandomWalk

class PinSAGESampler(RandomWalk):
    def __init__(self, sample_ratio, p_f=0.35, p_b=0.2, use_binomial_mean=False, **kwargs):
        super().__init__(sample_ratio, upper_bound=10, **kwargs)
        self.p = [p_f, p_b]
        self.use_binomial_mean = use_binomial_mean

    def _pinsage_sampler(self, g, available, init_nodes=None):
        n_samples = self.get_num_samples(g)
        sampled = torch.zeros(g.num_nodes())
        num_sampled = 0

        if init_nodes is not None:
            sampled[init_nodes] = 1
            available[init_nodes] = 0
            num_sampled = self._num_edges(g, sampled, num_sampled, n_samples)

        progress_bar = tqdm(disable=not self.verbose)
        sampled_nodes = torch.where(available == 1)[0]
        permutation = torch.randperm(len(sampled_nodes))
        sampled_nodes = sampled_nodes[permutation]
        index = 0
        while num_sampled < n_samples:

            ambassador = sampled_nodes[index]
            index += 1

            successors = g.successors(ambassador)

            new_node = (sampled[successors] == 0).any()

            # Update
            sampled[ambassador] = 1  # Sample user
            sampled[successors] = 1  # Sample items

            # Find edges
            if new_node:
                num_sampled = self._num_edges(g, sampled, num_sampled, n_samples)
                progress_bar.update(1)
                progress_bar.set_description(f'Sampled edges: {num_sampled}/{n_samples}')

        progress_bar.close()

        tg = g.subgraph(g.nodes()[sampled == 1], store_ids=True)
        sampled_eids = tg.edata[dgl.EID]

        return sampled_eids

    def initial_sample(self, g: dgl.DGLGraph, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        available = torch.zeros(g.num_nodes())  # not all nodes are available initially
        available[g.edges()[0]] = 1  # Select users only

        sampled_eids = self._pinsage_sampler(g, available, init_nodes)

        return sampled_eids

    def post_sampling(self, g: dgl.DGLGraph, sampled_nodes: torch.LongTensor, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        return super().post_sampling(g, sampled_nodes, init_nodes)

    def __str__(self):
        model_name = '{mname}_{sr}' if not self._str_arg else f'{{mname}}_{self._str_arg}_{{sr}}'
        return model_name.format(mname='pinsage', sr=self.sample_ratio)


if __name__ == '__main__':
    ps = PinSAGESampler(0.5, verbose=True)
    path = '../../amazon-book/ab_stratified/fold_0/train_cg_pos.dgl'
    g = dgl.load_graphs(path)[0][0]

    eids = ps.initial_sample(g)

    src, dst = g.find_edges(eids)
    nodes = torch.zeros(g.num_nodes()).to(torch.bool)
    nodes[dst] = 1
    path = '../../amazon-book/ab_stratified/fold_0/train_kg.dgl'
    kg = dgl.load_graphs(path)[0][0]
    nodes = ps.get_available_nodes(kg).to(torch.bool) & nodes
    eids = ps.post_sampling(kg, nodes.to(torch.float64))