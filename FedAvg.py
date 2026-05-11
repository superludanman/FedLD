import copy
import torch


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvgWeighted(weights, coeffs):
    if len(weights) != len(coeffs):
        raise ValueError("weights and coeffs must have same length")
    if len(weights) == 0:
        raise ValueError("weights cannot be empty")

    coeff_tensor = torch.tensor(coeffs, dtype=torch.float32)
    coeff_sum = float(coeff_tensor.sum().item())
    if coeff_sum <= 0:
        coeff_tensor = torch.ones_like(coeff_tensor) / len(coeffs)
    else:
        coeff_tensor = coeff_tensor / coeff_sum

    w_avg = copy.deepcopy(weights[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * coeff_tensor[0]
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k] * coeff_tensor[i]
    return w_avg
