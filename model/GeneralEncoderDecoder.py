from torch import nn


class GeneralEncoderDecoder(nn.Module):
    """
        General Encoder-Decoder Framework
    """
    def __init__(self, encoder, decoder, src_embed, tar_embed, generator):
        """
        init function
        :param encoder: Encoder module, convenient for change different encoder
        :param decoder: Decoder module, convenient for change different decoder
        :param src_embed: src language embedding
        :param tar_embed: target language embedding
        :param generator: generate the current word depends on Hidden states of Decoder
        """
        super(GeneralEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.src_embed = src_embed
        self.tar_embed = tar_embed

        self.generator = generator

    def forward(self, src, target, src_mask, tar_mask):
        return self.decode(self.encode(src, src_mask), src_mask, target, tar_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, target, tar_mask):
        return self.decoder(self.tar_embed(target), memory, src_mask, tar_mask)