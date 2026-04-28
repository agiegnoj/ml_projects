import torch
from torch.utils.data import Dataset, DataLoader
from DataSet import *
from Transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dl, vocab, padId, epochs):
    model = Transformer(vocab, 512, 8, 6, 120).to(device)
    model.fit(dl, padId, epochs)
    model.saveModel()

def buildVocab(sequences):
    vocab = set()
    for seq in sequences:
        vocab.update(seq)
    vocab = sorted(list(vocab))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    return vocab, stoi, itos

def numericalizeAndPad(seq, stoi, maxLen=maxSeqLength):
    ids = [stoi[tok] for tok in seq]
    if len(ids) > maxLen:
        ids = ids[:maxLen]
    else:
        ids += [stoi["EOS"]] * (maxLen - len(ids))
    return torch.tensor(ids, dtype=torch.long).to(device)

class ExpressionDataset(Dataset):
    def __init__(self, src, tgt, stoi):
        self.src = src
        self.tgt = tgt
        self.stoi = stoi

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        srcIds = numericalizeAndPad(self.src[idx], self.stoi)
        tgtIds = numericalizeAndPad(self.tgt[idx], self.stoi)
        return srcIds, tgtIds

def prepareData(datasetSize, batchSize):
    src, tgt = buildDataset(datasetSize)
    vocab, stoi, itos = buildVocab(src + tgt)
    padId = stoi["EOS"]
    sosId = stoi["SOS"]
    dataset = ExpressionDataset(src, tgt, stoi)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    return dataloader, vocab, stoi, itos, sosId, padId

if __name__ == "__main__":
    dl, vocab, stoi, itos, sosId, padId = prepareData(100000, 1000)
    train(dl, vocab, padId, 100)
