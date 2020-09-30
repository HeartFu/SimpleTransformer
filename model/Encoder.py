from torch import nn

from model import utils as utils
# from model.utils import clones, LayerNorm, SublayerConnection


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, layer, N):
        """
        Initialization
        :param layer: SubLayer -- subEncoder
        :param N: the number of subEncoder
        """
        super(Encoder, self).__init__()
        self.layers = utils.clones(layer, N)
        self.norm = utils.LayerNorm(layer.size)

    def forward(self, x, mask):
        # import pdb
        # pdb.set_trace()
        for layer in self.layers:
            x = layer(x, mask)
        output = self.norm(x)
        return output


class EncoderLayer(nn.Module):
    """
    EncoderLayer consists of self-attention and feed forward layer
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = utils.clones(utils.SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        z = lambda y: self.self_attn(y, y, y, mask)
        x = self.sublayer[0](x, z)
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
