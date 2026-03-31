import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, modelDim, numHeads):
        super().__init__()
        self.modelDim = modelDim
        self.numHeads = numHeads
        self.headDim = modelDim // numHeads

        self.Qmat = nn.Linear(self.modelDim, self.modelDim)
        self.Kmat = nn.Linear(self.modelDim, self.modelDim)
        self.Vmat = nn.Linear(self.modelDim, self.modelDim)
        self.Omat = nn.Linear(self.modelDim, self.modelDim)

    def forward(self, Q, K, V, mask):

        batchSize = Q.shape[0]
        seqLength = Q.shape[1]

        Q = self.Qmat(Q).view(batchSize, seqLength, self.numHeads, self.headDim).transpose(1, 2)
        K = self.Kmat(K).view(batchSize, seqLength, self.numHeads, self.headDim).transpose(1, 2)
        V = self.Vmat(V).view(batchSize, seqLength, self.numHeads, self.headDim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.headDim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V).transpose(1, 2).contiguous()
        attention = attention.view(batchSize, seqLength, self.modelDim)

        return self.Omat(attention)





