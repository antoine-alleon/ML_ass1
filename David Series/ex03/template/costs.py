# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss. You can calculate the loss using mse or mae."""
    # ***************************************************
    # Lost function by MSE 
    w = np.transpose(w); 
    N = len(y)
    losses = (((y - np.dot(tx,w))**2).mean(axis = 0)) / 2
    return losses