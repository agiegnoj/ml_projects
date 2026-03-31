import random
import re

import sympy
from sympy.core.backend import sympify
from tqdm import tqdm

maxDepth = 4

binOperators = ["**", "*", "/", "+", "-"]
functions = ["sin", "cos", "tan", "exp", "log", "sqrt", "asin", "acos", "atan"]
functionCount = len(functions)
maxIndex = len(binOperators) + len(functions) + 2
maxSeqLength = 120

integers = [str(i) for i in range(-9, 10)]

def modifyExpression(expr, next):
    newExp = ""

    if 1 <= next <= 5:
        operator = random.choice(binOperators)
        depth = random.choice([1, 2])
        expr2 = buildFunction(depth, maxIndex, "x", True)
        newExp = f"({expr})" + operator + f"({expr2})"
    elif next == 6:
        power = random.randint(-9, 9)
        while power == 0:
            power = random.randint(-9, 9)
        newExp = f"({expr})**{power}"
    elif next == 7:
        factor = random.randint(-9, 9)
        while factor == 0:
            factor = random.randint(-9, 9)
        newExp = f"({factor})*({expr})"
    elif 8 <= next <= maxIndex:
        function = random.choice(functions)
        newExp = f"{function}({expr})"

    return newExp


def buildFunction(maxDepth, maxIndex, expr, noAdditionalRecursion=False):
    for _ in range (maxDepth):
        if noAdditionalRecursion:
          nextOperation = random.randint(1, maxIndex)
        else:
          nextOperation = random.randint(6, maxIndex)
        expr = modifyExpression(expr, nextOperation)

    return expr if len(expr)<=maxSeqLength else buildFunction()

def tokenize(expr):
    expr = str(expr).strip()

    pattern = r"(?:\*\*|sin|cos|tan|exp|log|sqrt|asin|acos|atanh|x|[*/+\-()]|\d+)"
    tokens = re.findall(pattern, expr)
    tokens.insert(0, "SOS")
    while len(tokens) < maxSeqLength:
        tokens.append("EOS")

    return tokens

def parse(expr):
    return sympy.simplify(sympify(expr))

def derivative(expr):
    return sympy.diff(expr)


def buildDataset(size):
    src = []
    tgt = []

    for i in tqdm(range(size), desc="building the dataset"):
        function = buildFunction(random.randint(1, maxDepth), maxIndex, "x", False)
        if function != None:
            function = parse(str(function))
            der = derivative(function)
            src.append(tokenize(der))
            tgt.append(tokenize(function))

    return src, tgt
