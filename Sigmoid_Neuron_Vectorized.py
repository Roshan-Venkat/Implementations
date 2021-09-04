import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error

class Sigmoid:

  def __init__(self):
    self.w = None
    self.b = 0

  def fit(self, X ,Y , epoch = 100, learning_rate = 0.1, display_loss = False, initialize = True):
    if display_loss: 
      loss = {}
    if initialize:
      self.w = np.random.randn(1, X.shape[1])
      self.b = 0
    Y = Y.reshape(-1,1)

    for i in tqdm_notebook(range(epoch), total = epoch, unit = 'epoch'):
      dw = 0
      db = 0
      percep = np.dot(X,self.w.T) + self.b
      sigmoid = 1.0 / (1.0 + np.exp(-(percep), dtype=np.float128))
      
      dw = np.matmul(X.T, (sigmoid - Y) * sigmoid * (1 - sigmoid))
      dw = dw.T
      db = np.sum((sigmoid - Y) * sigmoid * (1 - sigmoid),axis = 0)
      self.w -= learning_rate * dw
      self.b -= learning_rate * db
      if display_loss:
        loss[i] = mean_squared_error(sigmoid,Y)


    if display_loss:
      plt.plot(list(loss.values()))
      plt.show()
