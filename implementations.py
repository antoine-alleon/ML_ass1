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
        e = y - tx.dot(w)
        losses = np.mean(np.abs(e))
    else:
    # ***************************************************
    # Lost function by MSE 
        losses = (((y - np.dot(tx,w))**2).mean(axis = 0)) / 2 #Loss calculated using MSE
    return losses

def gradient_descent(y, tx, initial_w, max_iters, gamma, ltype="MSE"):
    """Gradient descent algorithm."""
    ws = initial_w #initiate weights
    for n_iter in range(max_iters):
        
        gradient = compute_gradient (y, tx, ws) #compute the gradient for current weights

        # update w by gradient
        ws = ws - gamma*gradient #updates the new weights
    
    loss=compute_loss(y,tx,ws,ltype) #compute final error

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
    
'''def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):  OLD  
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

    return loss, ws'''

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    w = initial_w
    g = 0
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma*grad #computes the new w(t+1)
            
    loss = compute_loss(y, tx, w)
    return losses, ws

'''def least_squares(y, tx, n_max):  #ANTOINE
    """calculate the least squares solution."""
    xx=np.delete(tx,0,1) #save x vector
    for i in range(n_max-2):
        np.concatenate(tx,np.power(xx,i+2)) #add powers of x to tx
    w_opt=np.linalg.solve(np.matmul(np.transpose(tx),tx),np.dot(np.transpose(tx),y)) #compute weights
    loss=compute_loss(y,tx,w_opt) #compute error
    
    return loss, w_opt'''

def least_squares(y, tx): #DAVID
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_): #DAVID
    """implement ridge regression."""
    # ***************************************************
    # ridge regression: 
    tx_T = tx.transpose()
    a = np.dot(tx_T,tx) + 2*len(y)*lambda_*np.identity(tx.shape[1])
    b = np.dot(tx_T,y)
    return np.linalg.solve(a,b) #this way is 1 µs faster
    # ***************************************************
    
'''def ridge_regression(y, tx, n_max, lambda_): #ANTOINE
    """implement ridge regression."""
    lambda_=lambda_/(2*np.shape(y))
    xx=np.delete(tx,0,1) #save x vector
    for i in range(n_max-2):
        np.append(tx,np.power(xx,i+2)) #add powers of x to tx
    w_opt=np.linalg.solve(np.matmul(np.transpose(tx),tx)+lambda_*np.identity(np.shape(y)),np.dot(np.transpose(tx),y)) #compute weights
    loss=compute_loss(y,tx,w_opt) #compute error
    
    return loss,w_opt'''

################### ADDITIONNAL REGRESSIONS ##############

def compute_subgradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    #compute gradient and loss
    N = len(y)
    
    # MAE gradient: (demonstration is in notability ex6 - série 02)
    gradient = -((1/N)*(np.dot(np.transpose(tx),np.sign(y - np.dot(tx,w)))))    
    return gradient

def compute_stoch_subgradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # implement stochastic gradient computation.It's same as the gradient descent.
    return compute_subgradient(y, tx, w)
    
def stochastic_subgradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    w = initial_w
    g = 0
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            g = compute_stoch_subgradient(minibatch_y, minibatch_tx, w)        
            #update w by gradient
            w = w - gamma*g #computes the new w(t+1)
    losses = compute_loss(y, tx, w)

    return losses, w