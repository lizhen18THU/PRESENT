from .Utils import *

import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.distributions import Poisson

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    NoneType,
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax
)
from torch_geometric.utils.sparse import set_sparse_value

class GATConv(MessagePassing):
    r"""
    The graph attentional operator from the `"Graph Attention Networks" <https://arxiv.org/abs/1710.10903>`_ paper
    The code was built based on pyG framework: 'https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv'

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops (in case
            :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
class BayesianLinear(nn.Module):
    r"""
    Applies Bayesian Linear
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    """
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.weight_eps = None
            
#         self.reset_parameters()

    def reset_parameters(self, prior_mu, prior_log_sigma):
        # Initialization method of Adv-BNN
#         stdv = 1. / math.sqrt(self.weight_mu.size(1))
#         self.weight_mu.data.uniform_(-stdv, stdv)
#         self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        self.weight_mu.data = prior_mu
        self.weight_log_sigma.data = torch.ones_like(self.weight_log_sigma.data) * prior_log_sigma.to(self.weight_log_sigma.data.device)
        self.prior_mu = prior_mu
        self.prior_log_sigma = prior_log_sigma
        

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        
    def unfreeze(self) :
        self.weight_eps = None
    
    def _kld_loss(self, mu_0, log_sigma_0, mu_1, log_sigma_1):
        kl = log_sigma_1 - log_sigma_0 + \
        (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*torch.exp(log_sigma_1)**2) - 0.5
        return kl.mean()
    
    def bayesian_kld_loss(self):
        device = self.weight_mu.data.device
        return self._kld_loss(self.weight_mu, self.weight_log_sigma, self.prior_mu.to(device), self.prior_log_sigma.to(device))
    
    def forward(self, x):
        device = self.weight_mu.data.device
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma).to(device)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps.to(device)
            
        bias = None
            
        return F.linear(x, weight, bias)
    
class MLP_Module(nn.Module):
    def __init__(self, d_in, d_hid, d_out, norm="LayerNorm", activ="elu"):
        super().__init__()
        if activ == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activ == "gelu":
            self.activation = nn.GELU(approximate='tanh')
        elif activ == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        self.module_list = nn.ModuleList([nn.Linear(d_in, d_hid[0])])
        norm_layer = nn.LayerNorm(d_hid[0], eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d_hid[0], eps=1e-6)
        self.module_list.append(norm_layer)
        for i in range(1, len(d_hid)):
            self.module_list.append(nn.Linear(d_hid[i-1], d_hid[i]))
            norm_layer = nn.LayerNorm(d_hid[i], eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d_hid[i], eps=1e-6)
            self.module_list.append(norm_layer)
            
        self.output_layer = nn.Linear(d_hid[-1], d_out)
        
        
    def forward(self, x):
        for i in range(0, len(self.module_list), 2):
            x = self.module_list[i](x)
            x = self.module_list[i+1](x)
            x = self.activation(x)
        
        return self.output_layer(x)
    
class GATEncoder(nn.Module):
    def __init__(self, d_in, d_out, d_hid=(1024, 512), dropout=0.1, norm="BatchNorm", activ="relu"):
        super(GATEncoder, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        if activ=="elu":
            self.activation = nn.ELU(inplace=True)
        elif activ=="gelu":
            self.activation = nn.GELU(approximate="tanh")
        elif activ == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        self.enc_fc = nn.Linear(d_in, d_hid[0])
        self.enc_norm0 = nn.LayerNorm(d_hid[0], eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_hid[0], eps=1e-6)
        
        self.enc_gatconv1 = GATConv(in_channels=d_hid[0], out_channels=d_hid[1], dropout=dropout)
        self.enc_norm1 = nn.LayerNorm(d_hid[1], eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_hid[1], eps=1e-6)
        
        self.enc_gatconv2 = GATConv(in_channels=d_hid[1], out_channels=d_out, dropout=dropout)
        self.enc_norm2 = nn.LayerNorm(d_out, eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_out, eps=1e-6)

    def forward(self, x, edge_index):
        x = self.enc_fc(x)
        x = self.enc_norm0(x)
        x = self.activation(x)
        
        x = self.enc_gatconv1(x, edge_index, return_attention_weights=None)
        x = self.enc_norm1(x)
        x = self.activation(x)
        
        x = self.enc_gatconv2(x, edge_index, return_attention_weights=None)
        x = self.enc_norm2(x)
        
        return x
    
class BayesianGATEncoder(nn.Module):
    def __init__(self, d_in, d_out, d_hid=(1024, 512), dropout=0.1, norm="LayerNorm", activ="elu"):
        super(BayesianGATEncoder, self).__init__()
        
        self.d_prior = d_out//3
        self.d_in = d_in
        self.d_out = d_out
        
        if activ=="elu":
            self.activation = nn.ELU(inplace=True)
        elif activ=="gelu":
            self.activation = nn.GELU(approximate="tanh")
        elif activ == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
        self.bayesian_fc = BayesianLinear(d_in, self.d_prior)
        
        self.enc_fc = nn.Linear(d_in, d_hid[0])
        self.enc_norm1 = nn.LayerNorm(d_hid[0], eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_hid[0], eps=1e-6)
        
        self.enc_gatconv1 = GATConv(in_channels=d_hid[0], out_channels=d_hid[1], dropout=dropout)
        self.enc_norm2 = nn.LayerNorm(d_hid[1], eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_hid[1], eps=1e-6)
        
        self.enc_gatconv2 = GATConv(in_channels=d_hid[1], out_channels=d_out-self.d_prior, dropout=dropout)
        self.enc_norm3 = nn.LayerNorm(d_out, eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_out, eps=1e-6)
        
    def prior_initialize(self, prior, tight_factor=10):
        if not isinstance(prior, torch.FloatTensor):
            prior = torch.FloatTensor(prior)
        assert prior.shape[0] == self.d_prior, "prior weight dimension not match"
        
        prior_log_sigma = torch.log(prior.std()/tight_factor)
        self.bayesian_fc.reset_parameters(prior, prior_log_sigma)

    def bnn_loss(self):
        return self.bayesian_fc.bayesian_kld_loss()

    def freeze(self):
        self.bayesian_fc.freeze()
        
    def unfreeze(self):
        self.bayesian_fc.unfreeze()

    def forward(self, x, edge_index):
        bayesian_out = self.bayesian_fc(x)
        
        x = self.enc_fc(x)
        x = self.enc_norm1(x)
        x = self.activation(x)
        
        x = self.enc_gatconv1(x, edge_index, return_attention_weights=None)
        x = self.enc_norm2(x)
        x = self.activation(x)
        
        x = self.enc_gatconv2(x, edge_index, return_attention_weights=None)
        x = torch.cat([x, bayesian_out], -1)
        x = self.enc_norm3(x)
        
        return x

class BayesianMLPEncoder(nn.Module):
    def __init__(self, d_in, d_out, d_hid=(1024, 512), dropout=0.1, norm="LayerNorm", activ="elu"):
        super(BayesianMLPEncoder, self).__init__()
        
        self.d_prior = d_out//3
        self.d_in = d_in
        self.d_out = d_out
        
        if activ=="elu":
            self.activation = nn.ELU(inplace=True)
        elif activ=="gelu":
            self.activation = nn.GELU(approximate="tanh")
        elif activ == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
        self.bayesian_fc = BayesianLinear(d_in, self.d_prior)
        
        self.enc_fc = nn.Linear(d_in, d_hid[0])
        self.enc_norm1 = nn.LayerNorm(d_hid[0], eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_hid[0], eps=1e-6)
        
        self.enc_fc1 = nn.Linear(d_hid[0], d_hid[1])
        self.enc_norm2 = nn.LayerNorm(d_hid[1], eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_hid[1], eps=1e-6)
        
        self.enc_fc2 = nn.Linear(d_hid[1], d_out-self.d_prior)
        self.enc_norm3 = nn.LayerNorm(d_out, eps=1e-6) if norm=="LayerNorm" else nn.BatchNorm1d(d_out, eps=1e-6)
        
    def prior_initialize(self, prior, tight_factor=10):
        if not isinstance(prior, torch.FloatTensor):
            prior = torch.FloatTensor(prior)
        assert prior.shape[0] == self.d_prior, "prior weight dimension not match"
        
        prior_log_sigma = torch.log(prior.std()/tight_factor)
        self.bayesian_fc.reset_parameters(prior, prior_log_sigma)

    def bnn_loss(self):
        return self.bayesian_fc.bayesian_kld_loss()

    def freeze(self):
        self.bayesian_fc.freeze()
        
    def unfreeze(self):
        self.bayesian_fc.unfreeze()

    def forward(self, x):
        bayesian_out = self.bayesian_fc(x)
        
        x = self.enc_fc(x)
        x = self.enc_norm1(x)
        x = self.activation(x)
        
        x = self.enc_fc1(x)
        x = self.enc_norm2(x)
        x = self.activation(x)
        
        x = self.enc_fc2(x)
        x = torch.cat([x, bayesian_out], -1)
        x = self.enc_norm3(x)
        
        return x
     
class ZINBDecoder(nn.Module):
    def __init__(self, d_in, d_hid, d_out, zero_inflaten=True, is_recons=False, norm="LayerNorm", activ="elu", basic_module="Linear", dropout=0.1):
        super().__init__()
        self.zero_inflaten = zero_inflaten
        self.is_recons = is_recons
        self.basic_module = basic_module
        
        if activ == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activ == "gelu":
            self.activation = nn.GELU(approximate='tanh')
        elif activ == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
        self.module_list = nn.ModuleList([nn.Linear(d_in, d_hid[0])]) if basic_module == "Linear" else nn.ModuleList([GATConv(in_channels=d_in, out_channels=d_hid[0], dropout=dropout)])
        norm_layer = nn.LayerNorm(d_hid[0], eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d_hid[0], eps=1e-6)
        self.module_list.append(norm_layer)
        for i in range(1, len(d_hid)):
            if basic_module == "Linear":
                self.module_list.append(nn.Linear(d_hid[i-1], d_hid[i]))
            else:
                self.module_list.append(GATConv(in_channels=d_hid[i-1], out_channels=d_hid[i], dropout=dropout))
            norm_layer = nn.LayerNorm(d_hid[i], eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d_hid[i], eps=1e-6)
            self.module_list.append(norm_layer)
            
        self.pi_output = nn.Linear(d_hid[-1], d_out) if zero_inflaten else None
        self.disp_output = nn.Linear(d_hid[-1], d_out)
        self.mean_output = nn.Linear(d_hid[-1], d_out)
        self.expr_output = nn.Linear(d_hid[-1], d_out) if is_recons else None
        
    def forward(self, x, edge_index=None):
        for i in range(0, len(self.module_list), 2):
            x = self.module_list[i](x) if self.basic_module == "Linear" else self.module_list[i](x, edge_index)
            x = self.module_list[i+1](x)
            x = self.activation(x)
        
        pi_out = F.sigmoid(self.pi_output(x)) if self.zero_inflaten else None
        disp_out = torch.clamp(F.softplus(self.disp_output(x)), 1e-4, 1e4)
        mean_out = torch.clamp(F.softplus(self.mean_output(x)), 1e-5, 1e6)
        recons_expr = self.expr_output(x) if self.is_recons else None
        
        return pi_out, disp_out, mean_out, recons_expr

class ZIPDecoder(nn.Module):
    def __init__(self, d_in, d_hid, d_out, zero_inflaten=True, is_recons=False, norm="LayerNorm", activ="elu", basic_module="Linear", dropout=0.1):
        super().__init__()
        self.zero_inflaten = zero_inflaten
        self.is_recons = is_recons
        self.basic_module = basic_module
        
        if activ == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activ == "gelu":
            self.activation = nn.GELU(approximate='tanh')
        elif activ == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
        self.module_list = nn.ModuleList([nn.Linear(d_in, d_hid[0])]) if basic_module == "Linear" else nn.ModuleList([GATConv(in_channels=d_in, out_channels=d_hid[0], dropout=dropout)])
        norm_layer = nn.LayerNorm(d_hid[0], eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d_hid[0], eps=1e-6)
        self.module_list.append(norm_layer)
        for i in range(1, len(d_hid)):
            if basic_module == "Linear":
                self.module_list.append(nn.Linear(d_hid[i-1], d_hid[i]))
            else:
                self.module_list.append(GATConv(in_channels=d_hid[i-1], out_channels=d_hid[i], dropout=dropout))
            norm_layer = nn.LayerNorm(d_hid[i], eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d_hid[i], eps=1e-6)
            self.module_list.append(norm_layer)
        
        self.pi_output = nn.Linear(d_hid[-1], d_out) if zero_inflaten else None
        self.rho_output = nn.Linear(d_hid[-1], d_out)
        self.expr_output = nn.Linear(d_hid[-1], d_out) if is_recons else None
        self.peak_bias = nn.Parameter(torch.randn(1, d_out))
        
    def forward(self, x, edge_index=None):
        for i in range(0, len(self.module_list), 2):
            x = self.module_list[i](x) if self.basic_module == "Linear" else self.module_list[i](x, edge_index)
            x = self.module_list[i+1](x)
            x = self.activation(x)
        
        pi = F.sigmoid(self.pi_output(x)) if self.zero_inflaten else None
        rho = self.rho_output(x)
        omega = F.softmax(rho + self.peak_bias, dim=-1)
        recons_expr = self.expr_output(x) if self.is_recons else None
        
        return pi, None, omega, recons_expr
    
def ZINBLoss(x, pi, disp, mean, scale_factor=1.0, ridge_lambda=0.5):
    eps = 1e-10
    mean = mean * scale_factor
    if pi is not None:
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge

        return torch.mean(result)
    else:
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        result = t1 + t2
        
        return torch.mean(result)
    
def ZIPLoss(x, pi, omega, scale_factor=1.0, ridge_lambda=0.5):
    eps = 1e-10
    lamb = omega * scale_factor
    if pi is not None:
        po_case = -Poisson(lamb).log_prob(x) - torch.log(1.0-pi+eps)
        zero_po = torch.exp(-lamb)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_po) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, po_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        
        return torch.mean(result)
    
    else:
        result = -Poisson(lamb).log_prob(x)
        return torch.mean(result)
    
def NLL_loss(x, pi, param1, param2, scale_factor=1.0, ridge_lambda=0.5):
    if param1 is not None: ## ZINB distribution
        return ZINBLoss(x, pi, param1, param2, scale_factor, ridge_lambda)
    else: ## ZIP distribution
        return ZIPLoss(x, pi, param2, scale_factor, ridge_lambda)

def IOA_loss(rna_enc_out, cas_enc_out, adt_enc_out, ori_fused_out, tau=0.1, times=10, sample_size=None):
    ioa_loss = 0
    size = ori_fused_out.shape[0]
    if sample_size is None: sample_size = 5*size // times
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    num_omics = (rna_enc_out is not None) + (cas_enc_out is not None) + (adt_enc_out is not None)
    if num_omics < 1: return 0

    for i in range(times):
        if sample_size == "all" or sample_size > size:
            ori_fused_sampled = ori_fused_out
            rna_enc_sampled = rna_enc_out
            cas_enc_sampled = cas_enc_out
            adt_enc_sampled = adt_enc_out
        else:
            index = random.sample(list(range(size)), sample_size)
            ori_fused_sampled = ori_fused_out[index, :]
            rna_enc_sampled = rna_enc_out[index, :] if rna_enc_out is not None else None
            cas_enc_sampled = cas_enc_out[index, :] if cas_enc_out is not None else None
            adt_enc_sampled = adt_enc_out[index, :] if adt_enc_out is not None else None

        log_fused_distrib = F.log_softmax(torch.matmul(ori_fused_sampled, ori_fused_sampled.t()) / tau, dim=1)
        if rna_enc_out is not None:
            rna_enc_distrib = F.softmax(torch.matmul(rna_enc_sampled, rna_enc_sampled.t()) / tau, dim=1) 
            ioa_loss += kl_loss(log_fused_distrib, rna_enc_distrib) / num_omics
        if cas_enc_out is not None:
            cas_enc_distrib = F.softmax(torch.matmul(cas_enc_sampled, cas_enc_sampled.t()) / tau, dim=1)
            ioa_loss += kl_loss(log_fused_distrib, cas_enc_distrib) / num_omics
        if adt_enc_out is not None:
            adt_enc_distrib = F.softmax(torch.matmul(adt_enc_sampled, adt_enc_sampled.t()) / tau, dim=1)
            ioa_loss += kl_loss(log_fused_distrib, adt_enc_distrib) / num_omics
            
    return ioa_loss / times

### Contrastive learning: ref SimCLR mechanism to design IBA loss
def IBA_loss(xlat_batch, positive_indices, negative_indices, tau=0.1):
    loss = 0
    count = 0
    
    for i in range(xlat_batch.shape[0]):
        if positive_indices[i] is not None and negative_indices[i] is not None:
            above = torch.exp(F.cosine_similarity(xlat_batch[i:i+1], xlat_batch[positive_indices[i]]) / tau).sum()
            below = torch.exp(F.cosine_similarity(xlat_batch[i:i+1], xlat_batch[negative_indices[i]]) / tau).sum()
            loss += -torch.log(above / (above + below))
            count += 1
            
    if count>0: return loss / count
    else: return 0
    
def IBP_loss(xlat_batch, target_batch, batch_index, tau=0.1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    batches = torch.unique(batch_index)
    loss = 0
    count = 0
    
    for cur_batch in batches:
        idx = batch_index==cur_batch
        if idx.sum() > 1:
            cur_xlat = xlat_batch[idx]
            cur_target = target_batch[idx]
            log_xlat_distrib = F.log_softmax(torch.matmul(cur_xlat, cur_xlat.t()) / tau, dim=1)
            target_distrib = F.softmax(torch.matmul(cur_target, cur_target.t()) / tau, dim=1)
            loss += kl_loss(log_xlat_distrib, target_distrib)
            count += 1
            
    if count>0: return loss / count
    else: return 0