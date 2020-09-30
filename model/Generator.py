from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generate the word through the hidden states of Decoder
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
