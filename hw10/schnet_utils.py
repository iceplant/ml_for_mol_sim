######## DO NOT MODIFY THIS PART ######## 
import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor
from utils import MLP

class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(torch.nn.Module):
    def __init__(
        self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float
    ):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W



class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift
