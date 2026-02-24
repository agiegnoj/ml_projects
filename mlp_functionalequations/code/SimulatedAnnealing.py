import os
import numpy as np
from TrainAndEvaluate import *

def optimalHyperParameters( epochs=40, temperature = 1000, decay=0.93):
    """searches for better performing hyperparameters using simulated annealing algorithm"""
    # hiddenLayerdims, epochs, array of
    currentConfig = [100, 200, [1.0, 1.0, 1.0, 1.0]]
    optimalConfig = currentConfig

    minLoss = getLoss(currentConfig)
    saveConfigToFile(currentConfig, "SALogs.txt", minLoss)


    for e in range(epochs):
        print("epoch:", e)
        temperature *= decay
        newConfig = getNewConfig(currentConfig)

        newLoss = getLoss(newConfig)

        if newLoss <  minLoss:
            currentConfig = newConfig

            optimalConfig = newConfig
            saveConfigToFile(optimalConfig, "SALogs.txt",newLoss)
            minLoss = newLoss
        elif np.exp(-((newLoss-minLoss)/temperature)) > np.random.rand():
            currentConfig = newConfig


    return optimalConfig

def getLoss(newConfig):

    loss = 0.0
    loss += trainAndEvaluate(10000, newConfig[1], newConfig[0], 0, 1000, 'linear', newConfig[2][0], f1, False)
    loss += trainAndEvaluate(10000, newConfig[1], newConfig[0], -1, 1, 'exponential', newConfig[2][1], f2, False)
    loss += trainAndEvaluate(10000, newConfig[1], newConfig[0],  -1, 1, 'composition', newConfig[2][2], f4, False)
    loss += trainAndEvaluate(10000, newConfig[1], newConfig[0], -math.pi, math.pi, 'cosine', newConfig[2][3], f5, False)

    return loss

def getNewConfig(currentConfig, stepSizeDims = 20, stepSizeEpochs = 20,stepSizeDF = 0.05):
    newConfig = currentConfig

    newConfig[0] = np.random.randint(max(50, currentConfig[0]-stepSizeDims), min(400, currentConfig[0]+stepSizeDims))
    newConfig[1] =np.random.randint(max(50, currentConfig[0]-stepSizeEpochs), min(300, currentConfig[0]+stepSizeEpochs))
    for i in range (4):
        newConfig[2][i] = np.random.choice([max(0.05, newConfig[2][i]-stepSizeDF), min(2.2, newConfig[2][i]+stepSizeDF)])

    return newConfig


def saveConfigToFile(config, filename, score):
    with open(filename, "a") as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 40 + "\n")

        f.write(f"  Loss: {score}\n")
        f.write(f"  HiddenLayerDims: {config[0]}\n")
        f.write(f"  Epochs: {config[1]}\n")
        f.write(f"  Damping Factor Cauchy Linear: {config[2][0]}\n")
        f.write(f"  Damping Factor Cauchy Exponential: {config[2][1]}\n")
        f.write(f"  Damping Factor Composition: {config[2][2]}\n")
        f.write(f"  Damping Factor D´Alembert Trig: {config[2][3]}\n")


if __name__ == "__main__":
    if not os.path.exists("SALogs.txt"):
        with open("SALogs.txt", 'w') as file:
            file.write("Model optimization with simulated Annealing in 40 epochs\n")
    optimalHyperParameters(epochs=40, temperature=1000, decay=0.9)
