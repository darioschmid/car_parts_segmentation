import torch
import matplotlib.pyplot as plt
import numpy as np

def vector_distance(output,target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        batches = torch.norm(output-target,dim=1)
        return torch.mean(batches)
        

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / (pred.shape[0]*pred.shape[1]*pred.shape[2])

def car_part_accuracy(output,target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        mask = ((pred != 0) & (pred != 9))
        pred = pred[mask]
        target = target[mask]
        correct += torch.sum(pred == target).item()
    return correct / (torch.sum(mask).item() + 1)

def car_vs_background_accuracy(output,target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        mask = (target == 9)
        correct += torch.sum(pred[mask] == target[mask]).item()
    return correct / (torch.sum(mask).item())


# def destriminator_accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
