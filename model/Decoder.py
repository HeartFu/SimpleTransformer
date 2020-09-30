from torch import nn

from model import utils as utils

# from model.utils import clones, LayerNorm, SublayerConnection


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = utils.clones(layer, N)
        self.norm = utils.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tar_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tar_mask)

        return  self.norm(x)

class DecoderLayer(nn.Module):
    """
    EncoderLayer consists of self-attention, src-attention and feed forward layer
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = utils.clones(utils.SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tar_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tar_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
