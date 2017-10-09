# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from costs import *


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # compute loss for each combination of w0 and w1.
    
    for i in range(0, len(w0)):
        for j in range(0, len(w1)):
            w = [w0[i], w1[j]]; 
            cost = compute_loss(y, tx, w)
            losses[i, j] = cost;
            
    return losses

# ***************************************************