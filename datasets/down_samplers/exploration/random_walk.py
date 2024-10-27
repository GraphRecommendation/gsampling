import dgl
import torch
from tqdm import tqdm

from datasets.down_samplers.graph_sampler_base import GraphSamplerBase


class RandomWalk(GraphSamplerBase):
    def __init__(self, sample_ratio, num_walk=100, restart_probability=0.15, use_jumps=False, length=10, **kwargs):
        super().__init__(sample_ratio, **kwargs)
        self.num_walk = num_walk
        self.restart_probability = restart_probability
        self.use_jumps = use_jumps
        self.length = length

    def _random_walk(self, g, available, init_nodes=None):
        n_samples = self.get_num_samples(g)
        init_available = available.clone()

        with (g.local_scope()):
            g.ndata['available'] = available
            g.ndata['visited'] = torch.zeros(g.num_nodes())
            num_sampled = 0

            if init_nodes is not None:
                g.ndata['visited'][init_nodes] = 1
                g.ndata['available'][init_nodes] = 0
                num_sampled = self._num_edges(g, g.ndata['visited'], num_sampled, n_samples)

            n_walks = self.num_walk
            tot_walks = 0
            walks_per_restart = []
            n_restarts = -1
            progress_bar = tqdm(disable=not self.verbose)
            while num_sampled < n_samples:
                change = False
                if n_walks >= self.num_walk:
                    if tot_walks > 0:
                        walks_per_restart.append(tot_walks)
                    tot_walks = 0

                    if not (g.ndata['available'] == 1).any():
                        # Allow sampling from other nodes
                        g.ndata['available'] = self.get_available_nodes(g)
                        g.ndata['available'][init_available == 1] = 0

                    start_node = torch.multinomial(g.ndata['available'], 1)
                    start_node = start_node[0]
                    g.ndata['visited'][start_node] = 1
                    g.ndata['available'][start_node] = 0
                    n_walks = 0
                    n_restarts += 1
                    change = True

                current_node = start_node
                had_available_neighbors = False
                for _ in range(self.length):
                    if torch.rand(1) < self.restart_probability:
                        if self.use_jumps:
                            sample = torch.multinomial(init_available, 1)[0]
                        else:
                            sample = start_node
                    else:
                        neighbors = torch.cat([g.successors(current_node), g.predecessors(current_node)])
                        if len(neighbors) == 0:  # Stay at position until restart or jump
                            sample = current_node
                        else:
                            sample = neighbors[torch.randint(len(neighbors), (1,))][0]
                            had_available_neighbors = had_available_neighbors or (g.ndata['visited'][neighbors] == 0).any()

                    if (g.ndata['visited'][sample] == 0).any():
                        g.ndata['visited'][sample] = 1
                        change = True
                        n_walks = -1

                    current_node = sample

                n_walks += 1
                tot_walks += 1
                if change:
                    num_sampled = self._num_edges(g, g.ndata['visited'], num_sampled, n_samples)
                    progress_bar.update(1)
                    progress_bar.set_description(f'Sampled edges: {num_sampled}/{n_samples}, #start nodes: {n_restarts},'
                                                 f' #walks: {torch.mean(torch.FloatTensor(walks_per_restart)[-100:]).item():.2f}')
                # if entire walk without meeting new occurred (without jump to start node), restart
                elif not had_available_neighbors:
                    n_walks = self.num_walk

            progress_bar.close()
            tg = g.subgraph(g.nodes()[g.ndata['visited'] == 1])
            sampled_eids = tg.edata[dgl.EID]
        return sampled_eids

    def initial_sample(self, g: dgl.DGLGraph, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        available = self.get_available_nodes(g)

        sampled_eids = self._random_walk(g, available, init_nodes)

        return sampled_eids

    def post_sampling(self, g: dgl.DGLGraph, sampled_nodes: torch.LongTensor, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        sampled_eids = self._random_walk(g, sampled_nodes, init_nodes)

        return sampled_eids

    def __str__(self):
        model_name = '{mname}_{sr}' if not self._str_arg else f'{{mname}}_{self._str_arg}_{{sr}}'

        if self.use_jumps:
            return model_name.format(mname='random_jump', sr=self.sample_ratio)
        else:
            return model_name.format(mname='random_walk', sr=self.sample_ratio)


