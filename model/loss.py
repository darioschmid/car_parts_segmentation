import torch.nn.functional as F
import torch.nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def l1_loss(output, target):
    return F.l1_loss(output, target)



class crossentropy_loss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not  None:
            weight = torch.tensor(weight)
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, output, target):
        return self.loss(output, target)

class BCE_Loss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight).expand(256,256,-1).transpose(0,2)
        self.loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        
    def forward(self, output, target):
        return self.loss(output, target)

