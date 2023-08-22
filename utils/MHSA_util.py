"""Eunkwang Jeon, ViT-pytorch (2020)
Multi Head self Attention (MSA) block 

    Notes
    -----
    The original multi-head self-attention as described in https://arxiv.org/abs/1706.03762
    The following multi-head attention mechanism comes from https://arxiv.org/abs/2010.11929
    But we made some changes
    [GitHub repository](https://github.com/jeonsworld/ViT-pytorch)

    References
    ----------
    @article{dosovitskiy2020,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, 
    Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, 
    Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
"""  
import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import math
import torch
from torch import nn
from utils.util import LinearWithConstraint


#%%
class MultiHeadSelfAttention(nn.Module):
    '''
    Attention Module used to perform self-attention operation allowing the model to attend
    information from different representation subspaces on an input sequence of embeddings.
    The sequence of operations is as follows :-

    Input -> Query, Key, Value -> ReshapeHeads -> Query.TransposedKey -> Softmax -> Dropout
    -> AttentionScores.Value -> ReshapeHeadsBack -> Output

    Args:
        embed_dim: Dimension size of the hidden embedding
        heads: Number of parallel attention heads (Default=8)
        activation: Optional activation function to be applied to the input while
                    transforming to query, key and value matrixes (Default=None)
        dropout: Dropout value for the layer on attention_scores (Default=0.1)
        bias:  If specified, adds bias to input / output projection layers. Default: False

    Methods:
        _reshape_heads(inp) :- 
        Changes the input sequence embeddings to reduced dimension according to the number
        of attention heads to parallelize attention operation
        (batch_size, seq_len, embed_dim) -> (batch_size * heads, seq_len, reduced_dim)

        _reshape_heads_back(inp) :-
        Changes the reduced dimension due to parallel attention heads back to the original
        embedding size
        (batch_size * heads, seq_len, reduced_dim) -> (batch_size, seq_len, embed_dim)

        forward(inp) :-
        Performs the self-attention operation on the input sequence embedding.
        Returns the output of self-attention as well as atttention scores
        (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim), (batch_size * heads, seq_len, seq_len)

    Examples:
        >>> attention = Attention(embed_dim, heads, activation, dropout)
        >>> out, weights = attention(inp)
    '''
    def __init__(self, embed_dim, heads=8, activation='elu', dropout=0.1, bias=False, norm=.5):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.heads
        assert self.head_dim  * self.heads == self.embed_dim

        self.query = LinearWithConstraint(embed_dim, embed_dim, bias=bias, max_norm=norm)
        self.key = LinearWithConstraint(embed_dim, embed_dim, bias=bias, max_norm=norm)
        self.value = LinearWithConstraint(embed_dim, embed_dim, bias=bias, max_norm=norm)
        self.softmax = nn.Softmax(dim=-1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inp.size()
        assert embed_dim == self.embed_dim

        query = self.activation(self.query(inp))
        key   = self.activation(self.key(inp))
        value = self.activation(self.value(inp))

        # output of _reshape_heads(): (batch_size * heads, seq_len, reduced_dim) | reduced_dim = embed_dim // heads
        query = self._reshape_heads(query)
        key   = self._reshape_heads(key)
        value = self._reshape_heads(value)

        # attention_scores: (batch_size * heads, seq_len, seq_len) | Softmaxed along the last dimension
        A = query[0]
        B = key.transpose(1, 2)[0]
        C = torch.matmul(query, key.transpose(1, 2))
        attention_scores = self.softmax(torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.head_dim))

        # out: (batch_size * heads, seq_len, reduced_dim)
        out = torch.matmul(self.dropout(attention_scores), value)

        # output of _reshape_heads_back(): (batch_size, seq_len, embed_size)
        out = self._reshape_heads_back(out) # + inp

        return out, attention_scores

    def _reshape_heads(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inp.size()

        reduced_dim = self.embed_dim // self.heads

        out = inp.reshape(batch_size, seq_len, self.heads, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, seq_len, reduced_dim)

        # out: (batch_size * heads, seq_len, reduced_dim)
        return out

    def _reshape_heads_back(self, inp):
        # inp: (batch_size * heads, seq_len, reduced_dim) | reduced_dim = embed_dim // heads
        batch_size_mul_heads, seq_len, reduced_dim = inp.size()
        batch_size = batch_size_mul_heads // self.heads

        out = inp.reshape(batch_size, self.heads, seq_len, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        # out: (batch_size, seq_len, embed_dim)
        return out