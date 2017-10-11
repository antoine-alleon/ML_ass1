# -*- coding: utf-8 -*-
"""some initial data processing functions."""
import numpy as np


def load_data(fileName, sub_sample=True, add_outlier=False):
    """Load data and convert it to the metrics system."""
    path_dataset = fileName #nameofthefiletoimport 
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1) #, usecols=[1, 2]) #specifies which colomns we want to import (here it is 1 and 2, if nothing is written then all the table is imported. 
    #height = data[:, 0]
    #weight = data[:, 1]
    #gender = np.genfromtxt(
    #    path_dataset, delimiter=",", skip_header=1, usecols=[0],
    #    converters={0: lambda x: 0 if b"Male" in x else 1})
    
    # Convert to metric system
    #height *= 0.025
    #weight *= 0.454

    # sub-sample
    '''if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])'''

    #return height, weight, gender
    return data

def load_data_y(fileName):
    """Load data and convert it to the metrics system."""
    path_dataset = fileName #nameofthefiletoimport 
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1], dtype = str) #use only column 1 and import type str data
    return data

def format_data(x):
    num_samples=x.shape[0]
    x_mean=np.mean(x,axis=0)
    x=x-x_mean[:,None].T
    x_dev=np.std(x,axis=0)
    x=x/x_dev[:,None].T
    x=np.delete(x,[0,1],1)
    xp = np.c_[np.ones(num_samples), x]
    
    return xp

def format_data_y(x):
    y = np.array(range(len(x)))
    for i in range(len(x)):
        if (x[i]=="b"):
            y[i]=1
        else:
            y[i]=-1
    return y

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

'''
def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

'''
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]