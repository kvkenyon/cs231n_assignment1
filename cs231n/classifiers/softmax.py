from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # -correct_score + log(sum(e ** incorrect_class_score))
    num_train = y.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        
        # 1xC because X[i] 1xD and W DxC --> 1xC 
        f = X[i].dot(W)
            
        f -= np.max(f)
        Li = -f[y[i]]
        
        sum_ef = 0
        for j in range(num_classes):
            # Loss for class(j) 
            sum_ef += np.exp(f[j])
            
            # Analytic gradient
            if y[i] == j:
                sum_efj = np.sum(np.exp(f))
                dW.T[j] += ((np.exp(f[y[i]]) / sum_efj) - 1) * X[i]
                continue
                
            dW.T[j] += (np.exp(f[j]) /  np.sum(np.exp(f))) * X[i]
        Li += np.log(sum_ef)
        loss += Li

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    n = y.shape[0]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    f = X.dot(W) # X: NxD and W: DxC -> f: NxC
       
    p = softmax(f)

    loss = cross_entropy_loss(p, y)
    loss += reg * np.sum(W * W)

    # dW: DxC ; dW.T: CxD
    p[range(n), y] -= 1
    
    # each p[i,j] * X[i]
    dW = p[:,None] * X[:,:,None]
    dW = np.sum(dW, axis=0)
    
    dW /= n
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f), axis=1)[:,None]    
    
def cross_entropy_loss(p, y):
    # p: NxC
    # y: NX1
    n = y.shape[0]
    return np.mean(-np.log(p[range(n), y]))
    
    
    