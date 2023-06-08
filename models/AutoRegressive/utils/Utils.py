"""
Utility functions
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import Config


def haversine(c0, c1):
    """
    :param c0: coordinate 0 in form (lat0, lng0) with degree as unit
    :param c1: coordinate 1 in form (lat1, lng1) with degree as unit
    :return: The haversine distance of c0 and c1 in km
    Compute the haversine distance between
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    dLat = math.radians(c1[0] - c0[0])
    dLng = math.radians(c1[1] - c0[1])
    lat0 = math.radians(c0[0])
    lat1 = math.radians(c1[0])
    form0 = math.pow(math.sin(dLat / 2), 2)
    form1 = math.cos(lat0) * math.cos(lat1) * math.pow(math.sin(dLng / 2), 2)
    radius_of_earth = 6371  # km
    dist = 2 * radius_of_earth * math.asin(math.sqrt(form0 + form1))
    return dist


def batch2device(record, record_GD: dict, record_GCRN, query, target_G: torch.Tensor, target_D: torch.Tensor, device):
    """ Transfer all sample data into the device (cpu/gpu) """
    # Transfer record
    for temp_feat in Config.ALL_TEMP_FEAT_NAMES:
        if temp_feat != Config.LSTNET_TEMP_FEAT and record is not None:
            record[temp_feat] = [tuple([g.to(device) for g in gs]) for gs in record[temp_feat]]
        record_GD[temp_feat] = [(curD.to(device), curG.to(device)) for (curD, curG) in record_GD[temp_feat]]

    if record_GCRN is not None:
        record_GCRN = [tuple([g.to(device)]) for (g,) in record_GCRN]

    # Transfer query
    if query is not None:
        query = query.to(device)

    # Transfer target
    target_G = target_G.to(device)
    target_D = target_D.to(device)

    return record, record_GD, record_GCRN, query, target_G, target_D


def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=torch.Tensor([0])):
    """
    RMSE (Root Mean Squared Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values above the threshold are considered
    :return: RMSE-threshold, number of items considered
    """
    y_true_mask = y_true > threshold
    y_pred_filter = y_pred[y_true_mask]
    y_true_filter = y_true[y_true_mask]
    return torch.sum(torch.pow((y_true_filter - y_pred_filter), 2)), len(y_true_filter)


def MAE(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=torch.Tensor([0])):
    """
    MAE (Mean Absolute Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values above the threshold are considered (if threshold=3, result is MAE-3)
    :return: MAE-threshold, number of items considered
    """
    y_true_mask = y_true > threshold
    y_pred_filter = y_pred[y_true_mask]
    y_true_filter = y_true[y_true_mask]
    return torch.sum(torch.abs(y_true_filter - y_pred_filter)), len(y_true_filter)


def MAPE(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=torch.Tensor([0])):
    """
    MAPE (Mean Absolute Percentage Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values above the threshold are considered (if threshold=3, result is MAPE-3)
    :return: MAPE-threshold, number of items considered
    """
    y_true_mask = y_true > threshold
    y_pred_filter = y_pred[y_true_mask]
    y_true_filter = y_true[y_true_mask]
    # TODO: use EPSILON instead of 1
    return torch.sum(torch.abs((y_true_filter - y_pred_filter)/(y_true_filter + 1))), len(y_true_filter)


METRICS_FUNCTIONS_MAP = {
    'RMSE': RMSE,
    'MAPE': MAPE,
    'MAE': MAE,
}


# by RoshanRane in https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing/exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as "plot_grad_flow(model.named_parameters())" to
        visualize the gradient flow.
    """
    avg_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and (p.grad is not None) and ('bias' not in n):
            layers.append(n)
            avg_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    plt.figure(figsize=(7, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='c')
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.1, lw=1, color='b')
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color='k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation='vertical')
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.5)   # zoom in on the lower gradient regions
    plt.xlabel('Layers')
    plt.ylabel('Average Gradient')
    plt.title('Gradient flow')
    plt.grid(True)
    plt.legend([Line2D([0], [0], color='c', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(5)
    plt.ioff()


def genMetricsResStorage(num_metrics_threshold=len(Config.EVAL_METRICS_THRESHOLD_SET), tasks=Config.METRICS_FOR_WHAT):
    metrics_res = {}
    for metrics_for_what in tasks:
        metrics_res[metrics_for_what] = {}
        for metrics in METRICS_FUNCTIONS_MAP:
            metrics_res[metrics_for_what][metrics] = {'val': torch.zeros(num_metrics_threshold),
                                                      'num': torch.zeros(num_metrics_threshold)}
    return metrics_res


def aggrMetricsRes(metrics_res, metrics_thresholds, num_metrics_threshold, res_D, target_D, res_G, target_G):
    for mi in range(num_metrics_threshold):  # for the (mi)th threshold
        for metrics_for_what in metrics_res:
            curRes, curTar = (res_D, target_D) if metrics_for_what == 'Demand' else (res_G, target_G)
            for metrics in metrics_res[metrics_for_what]:
                curFunc = METRICS_FUNCTIONS_MAP[metrics]
                res, resN = curFunc(curRes, curTar, metrics_thresholds[mi])
                metrics_res[metrics_for_what][metrics]['val'][mi] += res.item()
                metrics_res[metrics_for_what][metrics]['num'][mi] += resN

    return metrics_res


def wrapMetricsRes(metrics_res):
    for metrics_for_what in metrics_res:
        for metrics in metrics_res[metrics_for_what]:
            metrics_res[metrics_for_what][metrics]['val'] /= metrics_res[metrics_for_what][metrics]['num']
            if metrics == 'RMSE':
                metrics_res[metrics_for_what][metrics]['val'] = torch.sqrt(metrics_res[metrics_for_what][metrics]['val'])

    return metrics_res


def evalMetrics(dataloader, eval_type, getResMethod, device, logr, *args):
    # Metrics with thresholds
    num_metrics_threshold = len(Config.EVAL_METRICS_THRESHOLD_SET)
    metrics_res = genMetricsResStorage(num_metrics_threshold=num_metrics_threshold, tasks=Config.METRICS_FOR_WHAT)
    metrics_thresholds = [torch.Tensor([threshold]) for threshold in Config.EVAL_METRICS_THRESHOLD_SET]
    if device:
        metrics_thresholds = [torch.Tensor([threshold]).to(device) for threshold in Config.EVAL_METRICS_THRESHOLD_SET]
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            # Clean GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            res_D, res_G, target_D, target_G = getResMethod(batch, device, args)

            metrics_res = aggrMetricsRes(metrics_res, metrics_thresholds, num_metrics_threshold,
                                         res_D, target_D, res_G, target_G)

        metrics_res = wrapMetricsRes(metrics_res)

        logr.log('> Metrics Evaluations for %s Set:\n' % eval_type)
        for metrics_for_what in metrics_res:
            logr.log('%s:\n' % metrics_for_what)
            for metrics in metrics_res[metrics_for_what]:
                for mi in range(num_metrics_threshold):
                    cur_threshold = Config.EVAL_METRICS_THRESHOLD_SET[mi]
                    logr.log('%s-%d = %.4f%s' % (metrics,
                                                 cur_threshold,
                                                 metrics_res[metrics_for_what][metrics]['val'][mi],
                                                 (', ' if mi != num_metrics_threshold - 1 else '\n')))

    return metrics_res


# Test
if __name__ == '__main__':
    print(haversine((40.4944, -74.2655), (40.9196, -73.6957)))  # 67.39581283189828
