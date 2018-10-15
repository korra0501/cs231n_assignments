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
  num_example = X.shape[0] # N
  num_class = W.shape[1] # C


  for i in range(num_example):
      # Compute score for each examples
      f_i = np.dot(X[i], W)

      # Normalize by subtracting examples with max value
      f_i -= np.amax(f_i)

      # Compute cost
      exp_scores = np.exp(f_i)
      probs = exp_scores / np.sum(exp_scores)
      loss += -np.log(probs[y[i]])

      # Compute gradient - VERY IMPORTANT!
      for k in range(num_class):
          dW[:,k] += (probs[k] - (k == y[i])) * X[i]

  # Compute average
  loss /= num_example # data loss
  dW /= num_example

  # Regularization
  loss += 0.5 * reg * np.sum(W * W) # regularization loss
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_example = X.shape[0] # N

  score = np.dot(X, W)
  score -= np.amax(score, axis=1, keepdims=True)

  exp_scores = np.exp(score)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  # query for the log probabilities assigned to
  # the correct classes in each example
  loss = np.sum(-np.log(probs[np.arange(num_example), y]))

  # Compute gradient
  ind = np.zeros_like(probs)
  ind[np.arange(num_example), y] = 1
  dW = np.dot(X.T, probs - ind)

  # Compute average
  loss /= num_example # data loss
  dW /= num_example

  # Regularization
  loss += 0.5 * reg * np.sum(W * W) # regularization loss
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
