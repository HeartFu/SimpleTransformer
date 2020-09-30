## Simple Transformer based on Pytorch

Transformer was proposed in the paper named "Attention is All You Need". It droped the CNN and RNN and use the attention module to build the language model which is based on Encoder-Decoder Framework.

### Pre-install libraries

```
pip intall torchtext spacy
python -m spacy download en
python -m spacy download de
```
It needs spacy library with English and German tokennization.

For Pytorch, I installed Pytorch-1.4.0. For install Pytorch, see the Pytorch Website.

### Train this model

```
python train.py
```

In this example, we used the IWSLT dataset from torchtext to train the German-English Translation task.

### Reference

1. The Annotated Transformer http://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
2. Transformer代码阅读 http://fancyerii.github.io/2019/03/09/transformer-codes/
3. Transformer图解 http://fancyerii.github.io/2019/03/09/transformer-illustrated/
4. The Illustrated Transformer http://jalammar.github.io/illustrated-transformer/