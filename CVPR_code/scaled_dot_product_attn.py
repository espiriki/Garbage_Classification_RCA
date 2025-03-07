import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, reverse=False, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.reverse = reverse

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature ** 0.5,
                            k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        if self.reverse:
            print("Reversing weights")
            dimension = attn.shape[-1]
            attn = (1.0-attn)/(dimension-1)

        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn