import math

import numpy as np
import torch

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from LossFunctions import *

def MLPRelu (n : int):
    """Simple MLP model"""
    MLP = nn.Sequential(
        nn.Linear(1, n, bias=True),
        nn.ReLU(),
        nn.Linear(n,n, bias=True),
        nn.ReLU(),
        nn.Linear(n, 1, bias=True)
    )
    return MLP

def MLPTanh (n : int):
    """Simple MLP model"""
    MLP = nn.Sequential(
        nn.Linear(1, n, bias=True),
        nn.Tanh(),
        nn.Linear(n,n, bias=True),
        nn.Tanh(),
        nn.Linear(n, 1, bias=True)
    )
    return MLP

def trainedMLP (model, epochs, dampingFactor, functionalEquation, trainingData):
    """train model indirectly on functional equation"""

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(trainingData, 1000, shuffle=True)

    if functionalEquation == 'linear':
      for i in tqdm(range (epochs)):
          for data in dataloader:
            inputX = data[0]
            inputY = data[1]
            fX = model(inputX)
            fY = model(inputY)
            fXPlusY = model(inputX + inputY)
            ffx = model(fX)

            loss = lossOne(fXPlusY, fX, fY, ffx, inputX, dampingFactor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    elif functionalEquation == 'exponential':
        for i in tqdm(range (epochs)):
            for data in dataloader:
             inputX = data[0]
             inputY = data[1]

             fXPlusY = model(inputX + inputY)
             fX = model(inputX)
             fY = model(inputY)
             f2 = model(torch.tensor([[2]], dtype=torch.float32))
             f0 = model(torch.tensor([[0]], dtype=torch.float32))

             loss = lossTwo(fXPlusY, fX, fY, f2, f0, dampingFactor)

             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

    elif functionalEquation == 'non solvable':
        epsilon = np.random.rand()*0.1
        for i in tqdm(range (epochs)):
            for data in dataloader:
             inputX = data[0]
             inputY = data[1]
             inputXPlusEpsilon = inputX+epsilon
             fXPlusEpsilon = model(inputXPlusEpsilon)

             fX = model(inputX)
             fY = model(inputY)
             fXPlusfY = model(inputX + fY)

             loss = lossThree(fX, fY, fXPlusfY, inputY, fXPlusEpsilon, dampingFactor)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

    elif functionalEquation == 'composition':
        for i in tqdm(range (epochs)):
            for data in dataloader:
             inputX = data[0]
             inputY = data[1]
             fXPlusY = model(inputX + inputY)
             fXMinusY = model(inputX - inputY)
             fX = model(inputX)
             fY = model(inputY)
             f1 = model(torch.tensor([[1]], dtype=torch.float32))

             loss = lossFour(fXPlusY, fXMinusY, fX, fY, f1, dampingFactor)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

    elif functionalEquation == 'cosine':
        for i in tqdm(range (epochs)):
            for data in dataloader:
                x = torch.tensor([[0.0]], requires_grad=True)
                y = model(x)
                inputX = data[0]
                inputY = data[1]

                fXPlusY = model(inputX + inputY)
                fXMinusY = model(inputX - inputY)

                fX, fY = model(inputX), model(inputY)

                dy_dx = torch.autograd.grad(
                    y, x,
                    grad_outputs=torch.ones_like(y),
                    create_graph=True
                )[0]

                d2y_dx2 = torch.autograd.grad(
                    dy_dx, x,
                    grad_outputs=torch.ones_like(dy_dx)
                )[0]

                fPI = model(torch.tensor([[math.pi]], dtype=torch.float32))

                loss = lossFive(fXPlusY, fXMinusY, fX, fY, d2y_dx2 , fPI, dampingFactor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model
