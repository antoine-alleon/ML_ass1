# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # Compute gradient and loss for MSE
    N = len(y)    
    # MSE 
    gradient = -((1/N)*(np.dot(np.transpose(tx), y - np.dot(tx,w))))
    cost = err = y - tx.dot(w)
    
    return gradient, cost

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # implement stochastic gradient computation.It's same as the gradient descent.
    return compute_gradient(y, tx, w)
    
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    ws = [initial_w]
    losses = []
    w = initial_w
    g = 0
    for n_iter in range(max_iters):
        #num_batches = 5
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(minibatch_y, minibatch_tx, w)

            # update w by gradient
            w = w - gamma*grad #computes the new w(t+1)
            loss = compute_loss(y, tx, w)
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws