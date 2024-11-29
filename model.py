# Imports 
import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange
import torch.nn.functional as F

# Generator
class Generator(nn.Module):
    def __init__(self, embdDim: int = 10, seqLen: int = 1024, latentZDim: int = 100,
                 channels: int = 3, patchSize: int = 15, depth: int = 3, nHeads: int = 5,
                 attnDropRate: float = 0.5, forwardDropRate: float = 0.5) -> None:
        super(Generator, self).__init__()
        # HyperParameters
        self.channels = channels
        self.latentZDim = latentZDim
        self.seqLen = seqLen
        self.embdDim = embdDim
        self.patchSize = patchSize
        self.depth = depth
        self.nHeads = nHeads
        self.attnDropRate = attnDropRate
        self.fwdDropRate = forwardDropRate
        # Input Block: we get the noise and use positional embeding the same way we do for
        # Transformer the only difference is generator has fake inputs but Trafo doesn't
        self.l1 = nn.Linear(self.latentZDim, self.seqLen * self.embdDim)
        self.positionalEmbd = nn.Parameter(torch.zeros(1, self.seqLen, self.embdDim))
        # Body of Generator
        self.blocks = GenTrafoEncoder(
            depth = self.depth,
            embdSize = self.embdDim,
            dropP = self.attnDropRate,
            fwdDropP = self.fwdDropRate,
            numHeads = self.nHeads
        )
        # Processing the output of Body
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embdDim, self.channels, 1, 1, 0)
        )

    def forward(self, Z: Tensor) -> Tensor:
        X = self.l1(Z).view(-1, self.seqLen, self.embdDim)
        X = X + self.positionalEmbd
        H, W = 1, self.seqLen
        X = self.blocks(X)
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        out = self.deconv(X.permute(0, 3, 1, 2))
        out = out.view(-1, self.channels, H, W)
        return out

class GenTrafoEncoder(nn.Sequential):
    def __init__(self, depth: int = 8, **kwargs):
        # You can apply the self attention for as many times as you want
        super().__init__(*[GenTrafoEncoderBlock(**kwargs) for _ in range(depth)])

class GenTrafoEncoderBlock(nn.Sequential):
    def __init__(self, embdSize: int, numHeads: int = 5, dropP: float = 0.5,
                 fwdDropP: float = 0.5, fwdExpansion: int = 4) -> None:

        super().__init__(
            # Block 1
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embdSize),
                MultiHeadAttention(embdSize, numHeads, dropP),
                nn.Dropout(dropP)
            )),
            # Block 2
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embdSize),
                FeedForwardBlock(embdSize, expansion = fwdExpansion, dropP = fwdDropP),
                nn.Dropout(dropP)
            )),
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
    def forward(self, X: Tensor, **kwargs) -> Tensor:
        res = X
        X = self.fn(X, **kwargs)
        X += res
        return X

class FeedForwardBlock(nn.Sequential):
    def __init__(self, embdSize: int, expansion: int, dropP: float) -> None:
        super().__init__(
            nn.Linear(embdSize, expansion * embdSize),
            nn.GELU(),
            nn.Dropout(dropP),
            nn.Linear(expansion * embdSize, embdSize),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, embdSize: int, numHeads: int, dropOut: float) -> None:
        super().__init__()
        self.embdSize = embdSize
        self.nHeads = numHeads
        self.keys = nn.Linear(embdSize, embdSize)
        self.queries = nn.Linear(embdSize, embdSize)
        self.values = nn.Linear(embdSize, embdSize)
        self.attnDrop = nn.Dropout(dropOut)
        self.projection = nn.Linear(embdSize, embdSize)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        # splitting to number of heads 
        keys = rearrange(self.keys(X), 'b n (h d) -> b h n d', h = self.nHeads)
        values = rearrange(self.values(X), 'b n (h d) -> b h n d', h = self.nHeads)
        queries = rearrange(self.queries(X), 'b n (h d) -> b h n d', h = self.nHeads)
        # we're summing key and query as Energy
        # batchSize, numHeads, queryLength, keyLen
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
        if mask is not None:
            fillValue = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fillValue)
        scaling = self.embdSize ** (1/ 2)
        att = F.softmax(energy / scaling, dim = -1)
        att = self.attnDrop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out
  
class DisTrafoEncoder(nn.Sequential):
    def __init__(self, depth: int = 8, **kwargs):
        # You can apply the self attention for as many times as you want
        super().__init__(*[GenTrafoEncoderBlock(**kwargs) for _ in range(depth)])

class DisTrafoEncoderBlock(nn.Sequential):
    def __init__(self, embdSize: int = 100, numHeads: int = 5, dropP: int = 0.,
                 fwdDropP: int = 0., fwdExpansion: int = 4) -> None:
        super().__init__(
            # Block 1
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embdSize),
                MultiHeadAttention(embdSize, numHeads, dropP),
                nn.Dropout(dropP)
            )),
            # Block 2
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embdSize),
                FeedForwardBlock(embdSize, expansion = fwdExpansion, dropP = fwdDropP),
                nn.Dropout(dropP)
            )),
        )

# This block is for our discriminator to classify fake from real 
class classificationHead(nn.Sequential):
    def __init__(self, embdSize: int = 100, nClasses: int = 2) -> None:
        super().__init__()
        self.clsHead = nn.Sequential(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(embdSize),
            nn.Linear(embdSize, nClasses)
        )
    
    def forward(self, X: Tensor) -> Tensor:
        out = self.clsHead(X)
        return out

class PatchEmbeddingLinear(nn.Module):
    def __init__(self, inChannels: int = 21, patchSize: int = 16, embdSize: int = 100,
                 seqLen: int = 1024) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = 1, s2 = patchSize),
            nn.Linear(patchSize * inChannels, embdSize)
        )
        self.clsToken = nn.Parameter(torch.randn(1, 1, embdSize))
        self.positions = nn.Parameter(torch.randn((seqLen // patchSize) + 1, embdSize))
        print(seqLen)

    def forward(self, X):
        b, _, _, _ = X.shape
        X = self.projection(X)
        print(X.shape)
        clsTokens = repeat(self.clsToken, '() n e -> b n e', b = b)
        print(clsTokens.shape)
        # prepend the cls token to the input
        X = torch.cat([clsTokens, X], dim = 1)
        print(X.shape)
        print(self.positions.shape)
        # add positional embeding
        X += self.positions
        return X   

class Discriminator(nn.Sequential):
    def __init__(self, inChannels: int = 21, patchSize: int = 16, embdDim: int = 100,
                 seqLen: int = 1024, depth: int = 3, nClasses: int = 2, **kwargs) -> None:
        super(Discriminator, self).__init__(
            PatchEmbeddingLinear(inChannels, patchSize, embdDim, seqLen),
            DisTrafoEncoder(depth, embdSize = embdDim, dropP = 0.5, fwdDropP = 0.5, 
                            **kwargs),
            classificationHead(embdDim, nClasses)
        )
