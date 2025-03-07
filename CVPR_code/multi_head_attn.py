''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from CVPR_code.scaled_dot_product_attn import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v,
                 reverse=False, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.reverse = reverse

        self.attention = \
            ScaledDotProductAttention(temperature=d_k, reverse=self.reverse)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual_input = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        mha_output, attn = self.attention(q, k, v, mask=mask)

        # b seq_len x d_model
        mha_output = mha_output.view(sz_b, len_q, self.d_model)

        # add and norm 1
        # LayerNorm(x + Sublayer(x))
        output_sub_layer_1 = self.layer_norm_1(residual_input + mha_output)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output_sub_layer_1 = \
            output_sub_layer_1.transpose(
                1, 2).contiguous().view(sz_b, len_q, -1)

        output_sub_layer_2 = self.dropout(self.fc(output_sub_layer_1))

        # add and norm 2
        # LayerNorm(x + Sublayer(x))
        output = self.layer_norm_2(output_sub_layer_1 + output_sub_layer_2)

        return output, attn
