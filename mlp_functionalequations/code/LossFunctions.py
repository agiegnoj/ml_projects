import torch

def lossOne(fXPlusY, fX, fY, ffx, inputX, dampingFactor):
    """train model on additivity, function iteration, positivity"""
    loss = torch.square(fXPlusY - (fX + fY)) + torch.square(ffx - 5*inputX) + dampingFactor*torch.relu(-fX)
    return loss.sum()

def lossTwo(fXPlusY, fX, fY, f2, f0, dampingFactor):
    """train model on particular solution of cauchys exponential equation"""
    loss = torch.square(fXPlusY - fX*fY).mean() + dampingFactor*torch.square(f2 - 67)+dampingFactor*torch.square(f0-1)
    return loss

def lossThree(fX, fY, fXPlusfY, inputY, fXPlusEpsilon, dampingFactor):
    """train model on a non solvable functional equation"""
    loss = torch.square(fXPlusfY-(fX+torch.square(inputY))).mean()+dampingFactor*torch.relu(fX-fXPlusEpsilon)
    return loss.sum()

def lossFour(fXPlusY, fXMinusY, fX, fY, f1, dampingFactor):
    loss = torch.square(fXPlusY * fXMinusY - torch.square(fX)*torch.square(fY)).mean()+dampingFactor*torch.square(f1-torch.e)
    return loss

def lossFive(fXPlusY, fXMinusY, fX, fY, d2y_d2x, fPI, dampingFactor):
    """train model on particular solution of D´Alemberts functional equation"""
    loss = torch.square((fXPlusY+fXMinusY)-2*fX*fY).mean()+ torch.relu(d2y_d2x)+ dampingFactor*torch.square(fPI+1)
    return loss


