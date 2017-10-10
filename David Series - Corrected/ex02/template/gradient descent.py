def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # Compute gradient and loss for MSE
    w = np.transpose(w)
    N = len(y)
    
    # MSE 
    gradient = -((1/N)*(np.dot(np.transpose(tx), y - np.dot(tx,w))))
    cost = err = y - tx.dot(w)
    
    return gradient, cost


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # compute gradient and loss
        gradient, cost = compute_gradient (y, tx, w)
        loss = calculate_mse(cost)

        # update w by gradient
        w = w - gamma*gradient #computes the new w(t+1)
        
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws