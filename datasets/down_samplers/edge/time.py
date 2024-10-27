import dgl
import torch
from tqdm import tqdm

from datasets.down_samplers.exploration import random_walk


class TimeSampler(random_walk.RandomWalk):
    def __init__(self, sample_ratio, reversed_=False, **kwargs):
        """
        Sample edges based on time.
        :param sample_ratio: number of samples to sample.
        :param reversed_: if True, sample oldest ratings first.
        """
        super().__init__(sample_ratio, **kwargs)
        self.reversed = reversed_

    def _time_sampler(self, g, available, init_nodes=None):
        n_samples = self.get_num_samples(g)

        # Reverse order is when we want to sample the oldest ratings first.
        sorted_edges = torch.argsort(g.edata['rating_time'], descending=not self.reversed)
        num_sampled = 0

        progress_bar = tqdm(disable=not self.verbose)
        with g.local_scope():
            g.ndata['visited'] = torch.zeros(g.num_nodes())
            g.edata['used'] = torch.zeros(g.num_edges())
            if init_nodes is not None:
                g.ndata['visited'][init_nodes] = 1
                num_sampled, eids = self._num_edges(g, g.ndata['visited'], num_sampled, n_samples, return_eids=True)
                g.edata['used'][eids] = 1

            start, end = 0, 0
            bz = 1
            length = len(sorted_edges)
            while num_sampled < n_samples and end < length:
                # available_edges = sorted_edges[g.edata['used'][sorted_edges] == 0]
                start, end = end, min(end + bz, length)
                src, dst = g.find_edges(sorted_edges[start:end])
                upper = max(min(1000, (n_samples - num_sampled) // 10), 1)
                if (g.ndata['visited'][src] == 0).any() or (g.ndata['visited'][dst] == 0).any():
                    g.ndata['visited'][src] = 1
                    g.ndata['visited'][dst] = 1
                    num_sampled, eids = self._num_edges(g, g.ndata['visited'], num_sampled, n_samples, return_eids=True)
                    g.edata['used'][eids] = 1
                    progress_bar.set_description(f'Sampled edges: {num_sampled}/{n_samples}, bz: {bz}')
                    progress_bar.update(1)
                    bz = int(min(max(1, bz - 1), upper))
                else:
                    bz = int(min(bz + 1, upper))

            tg = g.subgraph(g.nodes()[g.ndata['visited'] == 1])
            sampled_eids = tg.edata[dgl.EID]

        return sampled_eids

    def initial_sample(self, g: dgl.DGLGraph, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        available = torch.zeros(g.num_nodes())  # not all nodes are available initially
        available[torch.cat(g.edges())] = 1

        sampled_eids = self._time_sampler(g, available, init_nodes)

        return sampled_eids

    def post_sampling(self, g: dgl.DGLGraph, sampled_nodes: torch.LongTensor, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        return super().post_sampling(g, sampled_nodes, init_nodes)

    def __str__(self):
        model_name = '{mname}_{sr}' if not self._str_arg else f'{{mname}}_{self._str_arg}_{{sr}}'

        if self.reversed:
            return model_name.format(mname='reversed_time', sr=self.sample_ratio)
        else:
            return model_name.format(mname='time', sr=self.sample_ratio)