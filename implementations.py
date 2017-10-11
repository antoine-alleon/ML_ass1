# -*- coding: utf-8 -*-
#6 basic method implementations as described above in step 2: We want you to implement and use the methods 
#we have seen in class and in the labs. You will need to provide working implementations of the functions in Table 1. 
#If you have not finished them during the labs, you should start by implementing the first ones to have a working 
#toolbox before diving in the dataset.

#Return type: Note that all functions should return: (w, loss), which is the last weight vector of the method, 
#and the corresponding loss value (cost function). Note that while in previous labs you might have kept track of 
#all encountered w for iterative methods, here we only want the last one.

import numpy as np
import matplotlib.pyplot as plt

def compute_loss(y, tx, w, ltype="MSE"):
    """Calculate the loss.
    
    You can calculate the loss using mse or mae.
    """
    
    if (ltype=="MAE"):
    # ***************************************************
    # Lost function by MAE
        raise NotImplementedError
    else:
    # ***************************************************
    # Lost function by MSE 
        w = np.transpose(w); 
        N = len(y)
        losses = (((y - np.dot(tx,w))**2).mean(axis = 0)) / 2 #Loss calculated using MSE
    
    
    return losses

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    ws = initial_w #initiate weights
    for n_iter in range(max_iters):
        
        gradient = compute_gradient (y, tx, ws) #compute the gradient for current weights

        # update w by gradient
        
        ws = ws - gamma*gradient #updates the new weights
    
    loss=compute_loss(y,tx,ws) #compute final error

    return loss, ws

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    w = np.transpose(w)
    N = len(y)
    gradient = -((1/N)*(np.dot(np.transpose(tx), y - np.dot(tx,w))))
    
    return gradient

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    w = np.transpose(w)
    N = len(y)
    gradient = -((1/N)*(np.dot(np.transpose(tx), y - np.dot(tx,w))))
    return gradient
    
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    ws = initial_w
    g = 0
    for n_iter in range(max_iters):
        num_batches = 5
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            g = g + compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        g = g / num_batches #where g is the gradient         

        # update w by gradient
        ws = ws - gamma*g #computes the new w(t+1)
            
    loss=compute_loss(y,tx,ws)

    return loss, ws

def least_squares(y, tx, n_max):
    """calculate the least squares solution."""
    xx=np.delete(tx,0,1) #save x vector
    for i in range(n_max-2):
        np.concatenate(tx,np.power(xx,i+2)) #add powers of x to tx
    w_opt=np.linalg.solve(np.matmul(np.transpose(tx),tx),np.dot(np.transpose(tx),y)) #compute weights
    loss=compute_loss(y,tx,w_opt) #compute error
    
    return loss, w_opt

def ridge_regression(y, tx, n_max, lambda_):
    """implement ridge regression."""
    lambda_=lambda_/(2*np.shape(y))
    xx=np.delete(tx,0,1) #save x vector
    for i in range(n_max-2):
        np.append(tx,np.power(xx,i+2)) #add powers of x to tx
    w_opt=np.linalg.solve(np.matmul(np.transpose(tx),tx)+lambda_*np.identity(np.shape(y)),np.dot(np.transpose(tx),y)) #compute weights
    loss=compute_loss(y,tx,w_opt) #compute error
    
    return loss,w_opt