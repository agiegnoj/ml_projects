import random
from MLP import *
import logging
import datetime

logger = logging.getLogger(__name__)

def trainAndEvaluate(trainingDataSize : int, epochs : int, hiddenLayerDims : int, lowBnd:float, hiBnd : float, functionalEquation : str, dampingFactor : float, trueFunction, withLogs : bool):
    """Train models, evaluate them against the function they are supposed to approximate, get closed formula"""
    if withLogs:
       logger.info("\n"+functionalEquation)
    trainingData = getData(trainingDataSize, lowBnd, hiBnd)

    testData1 = torch.tensor([random.uniform(lowBnd, hiBnd) for _ in range(1000)], dtype=torch.float32)
    testData1 = testData1.view(-1, 1)

    modelRelu = trainedMLP(MLPRelu(hiddenLayerDims), epochs, dampingFactor, functionalEquation, trainingData)
    modelTanh = trainedMLP(MLPTanh(hiddenLayerDims), epochs, dampingFactor, functionalEquation, trainingData)

    modelRelu.eval()
    modelTanh.eval()

    with torch.no_grad():
        modelIResults = modelRelu(testData1)
        modelIResultsNormalized = torch.div(modelIResults, torch.max(modelIResults))
        modelIIResults = modelTanh(testData1)
        modelIIResultsNormalized = torch.div(modelIIResults, torch.max(modelIIResults))


    if functionalEquation != 'non solvable':
      expectedResults = trueFunction(testData1)
      expectedResultsNormalized = torch.div(expectedResults, torch.max(expectedResults))

      for i in range (10):
        x, mlpIX, mlpIIX, fX = float(testData1[i]), float(modelIResults[i]),  float(modelIIResults[i]), float(expectedResults[i])
        if withLogs:
          logger.info('x: ' + str(x) + '| mlpRelu(x): ' + str(mlpIX) + '| mlpTanh(x): '+ str(mlpIIX) + '| f(x):' + str(fX))

      modelILoss = float((abs(modelIResultsNormalized-expectedResultsNormalized)).mean())
      modelIILoss = float((abs(modelIIResultsNormalized-expectedResultsNormalized)).mean())

      if withLogs:
        logger.info('Relu MLP Loss  (|mlp(x)/max(mlp(x))-f(x)/(max(f(x))|/noOfTestSamples):' + str(modelILoss))
        logger.info( 'Tanh MLP loss (|mlp(x)/max(mlp(x))-f(x)/(max(f(x))|/noOfTestSamples):' + str(modelIILoss))

    elif functionalEquation == 'non solvable':
        epsilon = np.random.rand()*0.1
        testData2 = getData(1000, lowBnd, hiBnd)

        inputX = testData2[0]
        inputY = testData2[1]

        fX = modelRelu(inputX)
        inputXPlusEpsilon = inputX + epsilon
        fXPlusEpsilon = modelRelu(inputXPlusEpsilon)

        fY = modelRelu(inputY)
        fXPlusfY = modelRelu(inputX + fY)
        fXII = modelRelu(inputX)
        fXPlusEpsilonII = modelTanh(inputXPlusEpsilon)

        fYII = modelTanh(inputY)
        fXPlusfYII = modelTanh(inputX + fY)

        modelILoss = float(lossThree(fX, fY, fXPlusfY, inputY, fXPlusEpsilon, dampingFactor))/len(testData2)
        modelIILoss = float(lossThree(fXII, fYII, fXPlusfYII, inputY, fXPlusEpsilonII, dampingFactor))/len(testData2)

        if withLogs:
           logger.info('Relu MLP Loss  (FE Loss):' + str(modelILoss))
           logger.info('Tanh MLP loss (FE Loss):' + str(modelIILoss))

    return modelILoss + modelIILoss


def f1(x):
    return torch.sqrt(torch.tensor([5], dtype=torch.float32))*x

def f2(x):
    return torch.sqrt(torch.tensor([67], dtype=torch.float32))**x

def f4(x):
    return torch.exp(torch.square(x))

def f5(x):
    return torch.cos(x)

def getData(size, lowBnd, hiBnd):
    x = torch.rand(size, 1) * (hiBnd - lowBnd) + lowBnd
    y = torch.rand(size, 1) * (hiBnd - lowBnd) + lowBnd
    return x, y

if "__main__" == __name__:
    dt = datetime.datetime.now()
    logging.basicConfig(filename='feSolver.log', level=logging.INFO)
    logger.info('Started '+ str(dt))
    trainAndEvaluate(50000, 170, 131, 0, 1000, 'linear', 0.95, f1, True)
    trainAndEvaluate(50000, 170, 131, -1, 1, 'exponential', 1.15, f2, True)
    trainAndEvaluate(50000, 170, 131, -100, 100, 'non solvable', 1, None, True)
    trainAndEvaluate(50000, 170, 131, -1, 1, 'composition', 0.95, f4, True)
    trainAndEvaluate(50000, 170, 131, -math.pi, math.pi, 'cosine', 1.15, f5, True)
    logger.info('Finished\n')

