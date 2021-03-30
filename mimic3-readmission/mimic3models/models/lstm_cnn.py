from torch import nn
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import random

import mimic3models.metrics as m
import matplotlib.pyplot as plt


class LSTM_CNN(nn.Module):
    
    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1, bidirectional=False, dense=False):

        #dim, batch_norm, dropout, rec_dropout, task,
        #target_repl = False, deep_supervision = False, num_classes = 1,
        #depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = bidirectional
        self.dense = dense

        # some more parameters
        self.output_dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.dropout_words = 0.3
        self.dropout_rnn_U = 0.3
        self.drop_conv = 0.5

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layers,
                            dropout=self.rec_dropout,
                            bidirectional=self.bidirectional)

        # this is not in the original model
        self.act1 = nn.ReLU()

        self.do1 = nn.Dropout(self.dropout)
        self.cnn = nn.Conv1d()
        # concat the three outputs from the CNN
        self.do2 = nn.Dropout(self.drop_conv)
        self.dense = nn.Linear(self.hidden_dim, self.num_classes)

        # change linear layer inputs depending on if lstm is bidrectional
        #if not bidirectional:
        #    self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        #else:
        #    self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        #self.act2 = nn.ReLU()

        # change linear layer inputs depending on if lstm is bidrectional and extra dense layer isn't added
        if bidirectional and not dense:
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        else:
            self.final = nn.Linear(self.hidden_dim, 1)


    def forward(self, inputs, labels=None):
        out = inputs.unsqueeze(1)
        out, h = self.lstm(out)
        out = self.act1(out)
        #if self.dense:
        #    out = self.linear(out)
        #    out = self.act2(out)
        out = self.final(out)
        return out
    
    

class LSTM_CNN2(nn.Module):
    
    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1):

        #dim, batch_norm, dropout, rec_dropout, task,
        #target_repl = False, deep_supervision = False, num_classes = 1,
        #depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True
        #self.dense = dense

        # some more parameters
        #self.output_dim = dim
        #self.batch_norm = batch_norm
        self.dropout = 0.3
        self.rec_dropout = 0.3
        self.depth = lstm_layers
        self.drop_conv = 0.5
        self.num_classes = 1

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        if self.layers >=2:
            self.lstm1 = nn.LSTM(input_size=self.input_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.layers-1,
                                dropout=self.rec_dropout,
                                bidirectional=self.bidirectional,
                                batch_first=True)
            self.do0 = nn.Dropout(self.dropout)
            
        # this is not in the original model
        #self.act1 = nn.ReLU()
        if self.layers >=2:
            self.lstm2 = nn.LSTM(input_size=self.hidden_dim*2,
                                hidden_size=self.hidden_dim*2,
                                num_layers=1,
                                dropout=self.rec_dropout,
                                bidirectional=False,
                                batch_first=True)
        else:
            self.lstm2 = nn.LSTM(input_size=self.input_dim,
                                hidden_size=self.hidden_dim*2,
                                num_layers=1,
                                dropout=self.rec_dropout,
                                bidirectional=False,
                                batch_first=True)

        self.do1 = nn.Dropout(self.dropout)
        #self.bn0 = nn.BatchNorm1d(48 * self.hidden_dim*2)
        
        # three Convolutional Neural Networks with different kernel sizes
        nfilters=[2, 3, 4]
        nb_filters=100
        pooling_reps = []
        
        #self.cnns = nn.Module()
        #for idx, k in enumerate(nfilters):
        #    self.cnns.add_module(f"cnn{idx}", nn.Sequential(
        #        # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
        #        # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
        #        # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
        #        # padding_mode: str = 'zeros')
        #        nn.Conv1d(in_channels=16, out_channels=nb_filters, kernel_size=k,
        #                  stride=1, padding=0, dilation=1, groups=1, bias=True,
        #                  padding_mode='zeros'),
        #        nn.ReLU(),
        #        # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
        #        # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
        #        # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
        #        nn.MaxPool1d(kernel_size=2),
        #        nn.Flatten()
        #    ))
            
        self.cnn1 = nn.Sequential(
                # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
                # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
                # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
                # padding_mode: str = 'zeros')
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=2,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
                # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
                # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2300)
            )
        
        self.cnn2 = nn.Sequential(
                # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
                # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
                # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
                # padding_mode: str = 'zeros')
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=3,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
                # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
                # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2300)
            )
        
        self.cnn3 = nn.Sequential(
                # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
                # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
                # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
                # padding_mode: str = 'zeros')
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=4,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
                # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
                # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2200)
            )
        
        self.do2 = nn.Dropout(self.drop_conv)
        #self.act2 = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(6800)
        self.final = nn.Linear(6800, self.num_classes)
        #self.act3 = nn.ReLU()
        #self.act3 = nn.Sigmoid()


    def forward(self, inputs, labels=None):
        out = inputs #.unsqueeze(1)
        #print("inputs.shape = ", inputs.shape)
        if self.layers >=2:
            out, h = self.lstm1(out)
            out = self.do0(out)
        out, h = self.lstm2(out)
        #print("out lstm.shape = ", out.shape)
        ###out = self.act1(out) #[:,-1])
        #print("out relu.shape = ", out.shape)
        out = self.do1(out)
        
        #print("out do1.shape = ", out.shape)
        #out = self.bn0(out)
        
        pooling_reps = []
        #for cnn in model.cnns.children():
        #    pool_vecs = cnn(out.permute((0,2,1))) #.unsqueeze(dim=1))
        #    #print(f"out pool_vecs = ", pool_vecs.shape)
        #    pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn1(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn2(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn3(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
            
        #print("out pooling_reps = ", pooling_reps)
            
        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        out = self.do2(representation)
        #out = self.act2(out) ###
        #out = self.bn1(out) ###
        #out = self.act2(out)
        #print("out do2.shape = ", out.shape)
        out = self.final(out)
        #out = self.final(representation)
        #print("out final.shape = ", out.shape)
        #out = self.act3(out)
        #print("out final.shape = ", out.shape)
        return out
    
    
    
from typing import Optional, Tuple
import torch
from torch import nn, Tensor


class MCDualMixin:
    """Monte Carlo mixin
    This mixin provide a method `sample` to sample from defined model
    Use this Mixin by inheriting this class
    Assuming that model returns a tuple of 2 tensors"""

    def get_output_shape(self, *args):
        "Override this to get output dimensions."
        raise NotImplementedError("Need to define output shape")
    
    def sample(self, T:int, *args):
        # Construct empty outputs
        shape_m, shape_v = self.get_output_shape(*args)
        M, V = torch.empty(T, *shape_m), torch.empty(T, *shape_v)
        
        for t in range(T):
            M[t], V[t] = self(*args)
        
        return M, V


class MCSingleMixin:
    """Monte Carlo mixin
    This mixin provide a method `sample` to sample from defined model
    Use this Mixin by inheriting this class
    Assuming that model returns a single tensors"""

    def get_output_shape(self, *args):
        "Override this to get output dimensions."
        raise NotImplementedError("Need to define output shape")
    
    def sample(self, T:int, *args):
        # Construct empty outputs
        shape_m = self.get_output_shape(*args)
        M = torch.empty(T, *shape_m)
        
        for t in range(T):
            M[t] = self(*args)
        
        return M



class StochasticLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: Optional[float]=None):
        """
        Args:
        - dropout: should be between 0 and 1
        """
        super(StochasticLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if dropout is None:
            self.p_logit = nn.Parameter(torch.empty(1).normal_())
        elif not 0 < dropout < 1:
            raise Exception("Dropout rate should be between in (0, 1)")
        else:
            self.p_logit = dropout

        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Wg = nn.Linear(self.input_size, self.hidden_size)
        
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.init_weights()

    def init_weights(self):
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()
        
        self.Wi.weight.data.uniform_(-k,k)
        self.Wi.bias.data.uniform_(-k,k)
        
        self.Wf.weight.data.uniform_(-k,k)
        self.Wf.bias.data.uniform_(-k,k)
        
        self.Wo.weight.data.uniform_(-k,k)
        self.Wo.bias.data.uniform_(-k,k)
        
        self.Wg.weight.data.uniform_(-k,k)
        self.Wg.bias.data.uniform_(-k,k)
        
        self.Ui.weight.data.uniform_(-k,k)
        self.Ui.bias.data.uniform_(-k,k)
        
        self.Uf.weight.data.uniform_(-k,k)
        self.Uf.bias.data.uniform_(-k,k)
        
        self.Uo.weight.data.uniform_(-k,k)
        self.Uo.bias.data.uniform_(-k,k)
        
        self.Ug.weight.data.uniform_(-k,k)
        self.Ug.bias.data.uniform_(-k,k)
        
    # Note: value p_logit at infinity can cause numerical instability
    def _sample_mask(self, B):
        """Dropout masks for 4 gates, scale input by 1 / (1 - p)"""
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)
        GATES = 4
        eps = torch.tensor(1e-7)
        t = 1e-1
        
        ux = torch.rand(GATES, B, self.input_size)
        uh = torch.rand(GATES, B, self.hidden_size)

        if self.input_size == 1:
            zx = (1-torch.sigmoid((torch.log(eps) - torch.log(1+eps)
                                   + torch.log(ux+eps) - torch.log(1-ux+eps))
                                 / t))
        else:
            zx = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)
                                   + torch.log(ux+eps) - torch.log(1-ux+eps))
                                 / t)) / (1-p)
        zh = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)
                               + torch.log(uh+eps) - torch.log(1-uh+eps))
                             / t)) / (1-p)
        return zx, zh

    def regularizer(self):        
        if isinstance(self.p_logit, float):
            p = torch.tensor(self.p_logit)
        else:
            p = torch.sigmoid(self.p_logit)
        
        # Weight
        weight_sum = torch.tensor([
            torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("weight")
        ]).sum() / (1.-p)
        
        # Bias
        bias_sum = torch.tensor([
            torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("bias")
        ]).sum()
        
        if isinstance(self.p_logit, float):
            dropout_reg = torch.zeros(1)
        else:
             # Dropout
            dropout_reg = self.input_size * (p * torch.log(p) + (1-p)*torch.log(1-p))
        return weight_sum, bias_sum, 2.*dropout_reg
        
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        input shape (sequence, batch, input dimension)
        output shape (sequence, batch, output dimension)
        return output, (hidden_state, cell_state)
        """

        T, B = input.shape[0:2]

        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, dtype=input.dtype)
            c_t = torch.zeros(B, self.hidden_size, dtype=input.dtype)
        else:
            h_t, c_t = hx

        hn = torch.empty(T, B, self.hidden_size, dtype=input.dtype)

        # Masks
        zx, zh = self._sample_mask(B)
        
        for t in range(T):
            x_i, x_f, x_o, x_g = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_o, h_g = (h_t * zh_ for zh_ in zh)

            i = torch.sigmoid(self.Ui(h_i) + self.Wi(x_i))
            f = torch.sigmoid(self.Uf(h_f) + self.Wf(x_f))
            o = torch.sigmoid(self.Uo(h_o) + self.Wo(x_o))
            g = torch.tanh(self.Ug(h_g) + self.Wg(x_g))

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            hn[t] = h_t
        
        return hn, (h_t, c_t)


class StochasticLSTM(nn.Module):
    """LSTM stacked layers with dropout and MCMC"""

    def __init__(self, input_size: int, hidden_size: int, dropout:Optional[float]=None, num_layers: int=1):
        super(StochasticLSTM, self).__init__()
        self.num_layers = num_layers
        self.first_layer = StochasticLSTMCell(input_size, hidden_size, dropout)
        self.hidden_layers = nn.ModuleList([StochasticLSTMCell(hidden_size, hidden_size, dropout) for i in range(num_layers-1)])
    
    def regularizer(self):
        total_weight_reg, total_bias_reg, total_dropout_reg = self.first_layer.regularizer()
        for l in self.hidden_layers:
            weight, bias, dropout = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias
            total_dropout_reg += dropout
        return total_weight_reg, total_bias_reg, total_dropout_reg

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B = input.shape[1]
        h_n = torch.empty(self.num_layers, B, self.first_layer.hidden_size)
        c_n = torch.empty(self.num_layers, B, self.first_layer.hidden_size)
        
        outputs, (h, c) = self.first_layer(input, hx)
        h_n[0] = h
        c_n[0] = c

        for i, layer in enumerate(self.hidden_layers):
            outputs, (h, c) = layer(outputs, (h, c))
            h_n[i+1] = h
            c_n[i+1] = c

        return outputs, (h_n, c_n)
        
        
class LSTM_CNN3(nn.Module, MCSingleMixin):
    
    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1):

        super(LSTM_CNN3, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True
        #self.dense = dense

        # some more parameters
        #self.output_dim = dim
        #self.batch_norm = batch_norm
        self.dropout = 0.3
        self.rec_dropout = 0.3
        self.depth = lstm_layers
        self.drop_conv = 0.5
        self.num_classes = 1

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size

        self.lstm1 = StochasticLSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim*2,
                            dropout=self.rec_dropout,
                            num_layers=self.layers) #,
                            #bidirectional=self.bidirectional,
                            #batch_first=True)
        self.do0 = nn.Dropout(self.dropout)
        
        # three Convolutional Neural Networks with different kernel sizes
        nfilters=[2, 3, 4]
        nb_filters=100
        pooling_reps = []
            
        self.cnn1 = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=2,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2300)
            )
        
        self.cnn2 = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=3,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2300)
            )
        
        self.cnn3 = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=4,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2200)
            )
        
        self.do2 = nn.Dropout(self.drop_conv)
        #self.act2 = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(6800)
        self.final = nn.Linear(6800, self.num_classes)
        #self.act3 = nn.ReLU()
        #self.act3 = nn.Sigmoid()

    def regularizer(self):
        # Weight and bias regularizer
        weight_sum, bias_sum, dropout_reg = self.lstm1.regularizer()
        
        return weight_sum + bias_sum + dropout_reg

    def forward(self, inputs, labels=None):
        #out = inputs #.unsqueeze(1)
        out = inputs.permute((1,0,2)) # 0,1,2 -> 1,0,2
        #print("inputs.shape = ", inputs.shape)
        out, h = self.lstm1(out)
        out = self.do0(out)
        out = out.permute((1,0,2))
        
        pooling_reps = []
        #for cnn in model.cnns.children():
        #    pool_vecs = cnn(out.permute((0,2,1))) #.unsqueeze(dim=1))
        #    #print(f"out pool_vecs = ", pool_vecs.shape)
        #    pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn1(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn2(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn3(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
            
        #print("out pooling_reps = ", pooling_reps)
            
        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        out = self.do2(representation)
        #out = self.act2(out) ###
        #out = self.bn1(out) ###
        #out = self.act2(out)
        #print("out do2.shape = ", out.shape)
        out = self.final(out)
        #out = self.final(representation)
        #print("out final.shape = ", out.shape)
        #out = self.act3(out)
        #print("out final.shape = ", out.shape)
        return out


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LSTMNew(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        self.flatten_parameters() 
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class LSTM_CNN4(nn.Module):
    
    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1, dropout=0.3, dropout_w=0.3, dropout_conv=0.5):

        #dim, batch_norm, dropout, rec_dropout, task,
        #target_repl = False, deep_supervision = False, num_classes = 1,
        #depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN4, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True
        #self.dense = dense

        # some more parameters
        #self.output_dim = dim
        #self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = dropout_w
        self.depth = lstm_layers
        self.drop_conv = dropout_conv
        self.num_classes = 1

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        if self.layers >=2:
            self.lstm1 = LSTMNew(input_size=self.input_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.layers-1,
                                dropoutw=self.rec_dropout,
                                dropout=self.rec_dropout,
                                bidirectional=self.bidirectional,
                                batch_first=True)
            self.do0 = nn.Dropout(self.dropout)
            
        # this is not in the original model
        #self.act1 = nn.ReLU()
        if self.layers >=2:
            self.lstm2 = LSTMNew(input_size=self.hidden_dim*2,
                                hidden_size=self.hidden_dim*2,
                                num_layers=1,
                                dropoutw=self.rec_dropout,
                                dropout=self.rec_dropout,
                                bidirectional=False,
                                batch_first=True)
        else:
            self.lstm2 = LSTMNew(input_size=self.input_dim,
                                hidden_size=self.hidden_dim*2,
                                num_layers=1,
                                dropoutw=self.rec_dropout,
                                dropout=self.rec_dropout,
                                bidirectional=False,
                                batch_first=True)

        #self.do1 = nn.Dropout(self.dropout)
        #self.bn0 = nn.BatchNorm1d(48 * self.hidden_dim*2)
        
        # three Convolutional Neural Networks with different kernel sizes
        nfilters=[2, 3, 4]
        nb_filters=100
        pooling_reps = []
        
        #self.cnns = nn.Module()
        #for idx, k in enumerate(nfilters):
        #    self.cnns.add_module(f"cnn{idx}", nn.Sequential(
        #        # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
        #        # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
        #        # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
        #        # padding_mode: str = 'zeros')
        #        nn.Conv1d(in_channels=16, out_channels=nb_filters, kernel_size=k,
        #                  stride=1, padding=0, dilation=1, groups=1, bias=True,
        #                  padding_mode='zeros'),
        #        nn.ReLU(),
        #        # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
        #        # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
        #        # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
        #        nn.MaxPool1d(kernel_size=2),
        #        nn.Flatten()
        #    ))
            
        self.cnn1 = nn.Sequential(
                # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
                # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
                # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
                # padding_mode: str = 'zeros')
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=2,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
                # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
                # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2300)
            )
        
        self.cnn2 = nn.Sequential(
                # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
                # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
                # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
                # padding_mode: str = 'zeros')
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=3,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
                # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
                # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2300)
            )
        
        self.cnn3 = nn.Sequential(
                # torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]],
                # stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0,
                # dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True,
                # padding_mode: str = 'zeros')
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=4,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                # torch.nn.MaxPool1d(kernel_size: Union[T, Tuple[T, ...]],
                # stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0,
                # dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()#,
                #nn.BatchNorm1d(2200)
            )
        
        self.do2 = nn.Dropout(self.drop_conv)
        #self.act2 = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(6800)
        self.final = nn.Linear(6800, self.num_classes)
        #self.act3 = nn.ReLU()
        #self.act3 = nn.Sigmoid()


    def forward(self, inputs, labels=None):
        out = inputs #.unsqueeze(1)
        #print("inputs.shape = ", inputs.shape)
        if self.layers >=2:
            out, h = self.lstm1(out)
            out = self.do0(out)
        out, h = self.lstm2(out)
        #print("out lstm.shape = ", out.shape)
        ###out = self.act1(out) #[:,-1])
        #print("out relu.shape = ", out.shape)
        #out = self.do1(out)
        
        #print("out do1.shape = ", out.shape)
        #out = self.bn0(out)
        
        pooling_reps = []
        #for cnn in model.cnns.children():
        #    pool_vecs = cnn(out.permute((0,2,1))) #.unsqueeze(dim=1))
        #    #print(f"out pool_vecs = ", pool_vecs.shape)
        #    pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn1(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn2(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn3(out.permute((0,2,1)))
        #print(f"out pool_vecs = ", pool_vecs.shape)
        pooling_reps.append(pool_vecs)
            
        #print("out pooling_reps = ", pooling_reps)
            
        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        out = self.do2(representation)
        #out = self.act2(out) ###
        #out = self.bn1(out) ###
        #out = self.act2(out)
        #print("out do2.shape = ", out.shape)
        out = self.final(out)
        #out = self.final(representation)
        #print("out final.shape = ", out.shape)
        #out = self.act3(out)
        #print("out final.shape = ", out.shape)
        return out
    
# training loop of the LSTM model

def train(dataloader, model, optimizer, criterion, device):
    """
    main training function that trains model for one epoch/iteration cycle
    Args:
        :param dataloader: torch dataloader
        :param model: model to train
        :param optimizer: torch optimizer, e.g., adam, sgd, etc.
        :param criterion: torch loss, e.g., BCEWithLogitsLoss()
        :param device: the target device, "cuda" oder "cpu"
    """
    
    total_loss = []
    # initialize empty lists to store predictions and targets
    final_predictions = []
    final_targets = []
    
    # set model to training mode
    model.train()
    
    # iterate over batches from dataloader
    #for inputs, targets in tqdm(dataloader, desc="Train epoch"):
    for inputs, targets in dataloader:
        
        # set inputs and targets
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        
        # clear the gradients
        optimizer.zero_grad()
        
        # forward pass of inputs through the model
        predictions = model(inputs)
        
        # calculate the loss
        loss = criterion(predictions, targets.view(-1,1))
        #loss_ = loss + model.regularizer() / len(dataloader.dataset)
        
        total_loss.append(loss.item())
        # move predicitions and targets to list
        pred = predictions.detach().cpu().numpy().tolist()
        targ = targets.detach().cpu().numpy().tolist()
        final_predictions.extend(pred)
        final_targets.extend(targ)
        
        # compute gradienta of loss w.r.t. to trainable parameters of the model
        #loss_.backward()
        loss.backward()
        
        # single optimizer step
        optimizer.step()
        #if scheduler:
        #    scheduler.step()
        
    return total_loss, final_predictions, final_targets
        
def evaluate(dataloader, model, device):
    """
    main eval function
    Args:
        :param dataloader: torch dataloader for test data set
        :param model: model to evaluate
        :param device: the target device, "cuda" oder "cpu"
    """
    
    # initialize empty lists to store predictions and targets
    final_predictions = []
    final_targets = []
    
    # set model in eval mode
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        #for inputs, targets in tqdm(dataloader, desc="Eval epoch"):
        for inputs, targets in dataloader:
            # set inputs and targets
            #inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            
            # make predictions
            predictions = model(inputs)
            
            # move predicitions and targets to list
            predictions = predictions.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
            
    # return final predicitions and targets
    return final_predictions, final_targets


# trainer function
def trainer(dataloader_train, dataloader_val, modelclass=LSTM_CNN4, number_epochs=10, hidden_dim=16, lstm_layers=2, lr=1e-3,
            dropout=0.5, dropout_w=0.5, dropout_conv=0.5, best_loss=10000, best_accuracy=0, best_roc_auc=0, early_stopping=0,
            verbatim=False):
    
    if early_stopping == 0:
        early_stopping = number_epochs + 1
    early_stopping_counter = 0
    modelsignature = f"{number_epochs}_{hidden_dim}_{lstm_layers}_{lr}_{dropout}-{dropout_w}-{dropout_conv}"
    # create device depending which one is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fetch model
    model = modelclass(hidden_dim=hidden_dim, lstm_layers=lstm_layers,
                      dropout=dropout, dropout_w=dropout_w, dropout_conv=dropout_conv)

    # send model to device
    model.to(device)

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # initialize loss function
    loss = nn.BCEWithLogitsLoss()

    if verbatim:
        print("Training Model")
    train_loss_values = []
    val_loss_values = []

    # define threshold
    threshold = 0.5
    logit_threshold = torch.tensor (threshold / (1 - threshold)).log()

    for epoch in range(number_epochs):

        # train for one epoch
        error, outputs, targets = train(dataloader_train, model, optimizer, loss, device)
        train_loss_values.append(error)

        #y_pred = torch.sigmoid(torch.tensor(outputs))
        o = torch.tensor(outputs)

        #predicted_vals = y_pred > logit_threshold
        #o = np.where(outputs.clone().detach().numpy() > 0.5, 1., 0.)
        o = o > logit_threshold
        accuracy = metrics.accuracy_score(targets, o)
        #print(metrics.classification_report(targets, o))
        l = np.asarray(error)
        if verbatim:
            print(f"Epoch Train: {epoch}, Accuracy Score = {accuracy:.4f}, Loss = {l.mean():.4f}")
        #m.print_metrics_binary(targets,o.reshape(-1,))

        # validation of the model
        outputs, targets = evaluate(dataloader_val, model, device)

        #y_pred = torch.sigmoid(torch.tensor(outputs))
        outputs = torch.tensor(outputs)

        #predicted_vals = y_pred > logit_threshold
        o = outputs > logit_threshold
        accuracy = metrics.accuracy_score(targets, o)
        #print(metrics.classification_report(targets, o))
        #l = np.asarray(error)
        l = nn.BCEWithLogitsLoss()(outputs, torch.tensor(targets).detach().view(-1,1))
        val_loss_values.append(l)

        fpr, tpr, threshold = metrics.roc_curve(targets, outputs)
        roc_auc = metrics.auc(fpr, tpr)
        if verbatim:
            print(f"Epoch Val: {epoch}, Accuracy Score = {accuracy:.4f} ({best_accuracy:.4f}), ROCAUC = {roc_auc:.4f} ({best_roc_auc:.4f}), Loss = {l.mean():.4f} ({best_loss:.4f})")
            print("-"*20)
        #m.print_metrics_binary(targets, o.reshape(-1,))
        
        scheduler.step(roc_auc)

        if l < best_loss:
            best_loss = l
            # save model
            if verbatim:
                print("Saving model for best Loss...")
            torch.save(model.state_dict(), "./model_loss.pth")
            torch.save(model.state_dict(), f"./model__{modelsignature}__epoch-{epoch}_loss-{l}_acc-{accuracy}_auc-{roc_auc}.pth")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            # save model
            if verbatim:
                print("Saving model for ROC AUC...")
            early_stopping_counter = 0
            torch.save(model.state_dict(), "./model_roc_auc.pth")
            torch.save(model.state_dict(), f"./model__{modelsignature}__epoch-{epoch}_loss-{l}_acc-{accuracy}_auc-{roc_auc}.pth")
        else:
            early_stopping_counter += 1

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # save model
            if verbatim:
                print("Saving model...")
            torch.save(model.state_dict(), "./model_best.pth")
            torch.save(model.state_dict(), f"./model__{modelsignature}__epoch-{epoch}_loss-{l}_acc-{accuracy}_auc-{roc_auc}.pth")
        
        if early_stopping_counter > early_stopping:
            if verbatim:
                print("Early stopping done.")
            break
    
    return (best_loss, best_accuracy, best_roc_auc), train_loss_values, val_loss_values, modelsignature


def calcMetrics(model, dataloader_test, filename, title):
    # define threshold
    threshold = 0.5
    logit_threshold = torch.tensor (threshold / (1 - threshold)).log()
    device = next(model.parameters()).device

    model.load_state_dict(torch.load(filename))
    model.eval()
    
    print()
    print(title)
    print("=" * len(title))

    # validation of the model
    outputs, targets = evaluate(dataloader_test, model, device)

    #y_pred = torch.sigmoid(torch.tensor(outputs))
    outputs = torch.tensor(outputs)

    o = outputs > logit_threshold
    accuracy = metrics.accuracy_score(targets, o)
    print(metrics.classification_report(targets, o))

    l = nn.BCEWithLogitsLoss()(outputs, torch.tensor(targets).detach().view(-1,1))

    print(f"Accuracy Score = {accuracy}, Loss = {l.mean()}")
    print("-"*20)
    m.print_metrics_binary(targets, outputs.reshape(-1,))

    fpr, tpr, threshold = metrics.roc_curve(targets, outputs)
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC AUC = ", roc_auc)
    return roc_auc, targets, outputs

def plotLoss(train_loss, val_loss):
    def rollavg_direct(a,n): 
        assert n%2==1
        b = a*0.0
        for i in range(len(a)) :
            b[i]=a[max(i-n//2,0):min(i+n//2+1,len(a))].mean()
        return b

    plt.figure(figsize=(10,10))
    plt.title('Train/Val Loss')
    plt.plot([np.asarray(l).mean() for l in train_loss], label="Train loss")
    plt.plot([np.asarray(l).mean() for l in val_loss], label="Val loss")
    plt.plot(rollavg_direct(np.asarray([np.asarray(l).mean() for l in val_loss]),21))

    plt.legend(loc = 'upper right')
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.show()
    
def plotAUC(targets, outputs):
    fpr, tpr, threshold = metrics.roc_curve(targets, outputs)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()