from abc import ABC, abstractmethod
import random

import dgl
import numpy as np
import torch
from torch import Tensor


class GraphSamplerBase(ABC):
    def __init__(self, sample_ratio, upper_bound=100, str_arg='', verbose=False):
        self.sample_ratio = sample_ratio
        self.verbose = verbose
        self._str_arg = str_arg
        self._history = 5
        self._lower_bound = upper_bound
        self._cur_iteration = 0
        self._next_eval_step = 0
        self._samples_per_eval = []
        self._eval_steps = []
        self.reset()

    def reset(self):
        self._cur_iteration = 0
        self._next_eval_step = 0
        self._samples_per_eval = []
        self._eval_steps = []

    @abstractmethod
    def initial_sample(self, g: dgl.DGLGraph, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        pass

    @abstractmethod
    def post_sampling(self, g: dgl.DGLGraph, sampled_nodes: torch.LongTensor, init_nodes: torch.LongTensor = None) -> dgl.DGLGraph:
        pass

    def _num_edges(self, g: dgl.DGLGraph, selected: torch.LongTensor, n_edges: int, required_n_edges: int, return_eids=False) -> int:
        eids = []
        if self._cur_iteration == self._next_eval_step:
            tg = g.subgraph(g.nodes()[selected == 1], store_ids=True)
            n_edges = tg.num_edges()
            eids = tg.edata[dgl.EID]
            self._samples_per_eval.append(n_edges)
            self._eval_steps.append(self._cur_iteration)

            # after sufficient
            if len(self._samples_per_eval) < self._history:
                self._next_eval_step += 1
            else:
                length = len(self._samples_per_eval)
                new_samples = [(self._samples_per_eval[i] - self._samples_per_eval[i-1]) /
                               (self._eval_steps[i] - self._eval_steps[i-1])
                               for i in range(length-self._history, length)]
                m = np.mean(new_samples)
                steps_to_completion = max(1, min(int((required_n_edges - n_edges) // m) // 5, self._lower_bound))
                self._next_eval_step += steps_to_completion

        self._cur_iteration += 1
        if return_eids:
            return n_edges, eids
        else:
            return n_edges

    def get_num_samples(self, g: dgl.DGLGraph) -> int:
        return g.num_edges() * self.sample_ratio

    @staticmethod
    def get_available_nodes(g: dgl.DGLGraph) -> Tensor:
        available = torch.zeros(g.num_nodes())
        available[torch.cat(g.edges())] = 1
        return available

    def seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        dgl.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def __str__(self):
        pass
