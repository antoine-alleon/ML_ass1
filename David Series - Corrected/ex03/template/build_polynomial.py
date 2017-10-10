# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # polynomial basis function:
    phi_mat = np.empty([len(x),degree+1])
    for i in range(len(x)):
        for j in range(degree+1):
            phi_mat[i,j] = x[i]**j
                
    return phi_mat
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************