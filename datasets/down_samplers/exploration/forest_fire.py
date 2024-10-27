import dgl
import torch
from tqdm import tqdm

from datasets.down_samplers.graph_sampler_base import GraphSamplerBase

class ForestFire(GraphSamplerBase):
    """
    https://dl-acm-org.zorac.aub.aau.dk/doi/pdf/10.1145/1081870.1081893
    """

    def __init__(self, sample_ratio, p_f=0.35, p_b=0.2, use_binomial_mean=False, **kwargs):
        super().__init__(sample_ratio, **kwargs)
        self.p = [p_f, p_b]
        self.use_binomial_mean = use_binomial_mean

    def _forest_fire(self, g, available, init_nodes=None):
        n_samples = self.get_num_samples(g)
        burning = torch.zeros(g.num_nodes())
        frontier = torch.zeros(g.num_nodes())
        num_sampled = 0
        init_available = available.clone()

        if init_nodes is not None:
            burning[init_nodes] = 1
            available[init_nodes] = 0
            num_sampled = self._num_edges(g, burning, num_sampled, n_samples)

        progress_bar = tqdm(disable=not self.verbose)
        n_restarts = 0
        while num_sampled < n_samples:
            if not frontier.any():
                if not (available == 1).any():
                    # Allow sampling from other nodes
                    available = self.get_available_nodes(g)
                    available[init_available == 1] = 0

                ambassador = torch.multinomial(available, 1)[0]
                if num_sampled == 0:
                    neighbors = torch.cat([g.predecessors(ambassador), g.successors(ambassador)])
                    w = neighbors[torch.randint(len(neighbors), (1,))][0]
                else:
                    w = ambassador

                # Already sampled
                if burning[w] == 1:
                    continue
                else:
                    n_restarts += 1
                    burning[w] = 1
                    available[w] = 0
                    frontier[w] = 1

            w = torch.multinomial(frontier, 1)[0]

            # Remove from frontier
            frontier[w] = 0

            # Get neighbors
            in_neigh, out_neigh = g.predecessors(w), g.successors(w)

            any_burned = False
            for neigh, p in zip([out_neigh, in_neigh], self.p):
                if len(neigh) == 0:
                    continue

                # Sample mean should be (1 - self.p)^-1, thus we sample as follows:
                if self.use_binomial_mean:
                    prob = min(1., ((1 - p) ** -1) / len(neigh))
                else:
                    prob = min(1, p)
                b = torch.distributions.Bernoulli(probs=prob)
                samples = b.sample((len(neigh),)).to(torch.bool)

                neigh = neigh[samples]  # select only sampled nodes
                non_burned = neigh[burning[neigh] != 1]
                frontier[non_burned] = 1  # add non-burning nodes to frontier
                burning[neigh] = 1  # add to burning
                available[neigh] = 0  # remove from available

                any_burned = any_burned or len(non_burned) > 0

            # Find edges
            if any_burned:
                num_sampled = self._num_edges(g, burning, num_sampled, n_samples)
                progress_bar.update(1)
                progress_bar.set_description(f'Sampled edges: {num_sampled}/{n_samples}, burning patches: {n_restarts}')

        progress_bar.close()

        tg = g.subgraph(g.nodes()[burning == 1], store_ids=True)
        sampled_eids = tg.edata[dgl.EID]

        return sampled_eids

    def initial_sample(self, g: dgl.DGLGraph, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        available = torch.zeros(g.num_nodes())  # not all nodes are available initially
        available[torch.cat(g.edges())] = 1

        sampled_eids = self._forest_fire(g, available, init_nodes)

        return sampled_eids

    def post_sampling(self, g: dgl.DGLGraph, sampled_nodes: torch.LongTensor, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        sampled_eids = self._forest_fire(g, sampled_nodes, init_nodes)

        return sampled_eids

    def __str__(self):
        model_name = '{mname}_{sr}' if not self._str_arg else f'{{mname}}_{self._str_arg}_{{sr}}'
        if self.use_binomial_mean:
            return model_name.format(mname='forest_fire_binomial', sr=self.sample_ratio)
        else:
            return model_name.format(mname='forest_fire', sr=self.sample_ratio)


if __name__ == '__main__':
    ff = ForestFire(0.5, p_f=0.9, p_b=0.8, verbose=True, use_binomial_mean=True)
    path = '../../yelpkg/yk_stratified/fold_0/train_ckg_rev.dgl'
    g = dgl.load_graphs(path)[0][0]

    eids = ff.initial_sample(g)

    src, dst = g.find_edges(eids)
    nodes = torch.zeros(g.num_nodes()).to(torch.bool)
    nodes[dst] = 1
    path = '../../yelpkg/yk_stratified/fold_0/train_kg.dgl'
    kg = dgl.load_graphs(path)[0][0]
    nodes = ff.get_available_nodes(kg).to(torch.bool) & nodes
    eids = ff.post_sampling(kg, nodes.to(torch.float64))