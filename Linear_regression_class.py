class Linear_regression:
    def __init__(self):
        self.m = 0
        self.b = 0
        self.n = 0
    def Line(self,X):
        return self.m * X + self.b

    def error(self,Y,Y_pred):
        return Y - Y_pred

    def grad_M(self,X,Y):
        return (-2 / self.n) * np.dot(X, self.error(Y, self.Line(X)))

    def grad_B(self,X,Y):
        return (-2 / self.n) * np.sum(self.error(Y, self.Line(X)))

    def fit(self,X, Y, m, b, epoch= 1000, L= 0.0001):
        self.n = len(X)
        for i in range(epoch):
            self.m = self.m - L * self.grad_M(X,Y)
            self.b = self.b - L * self.grad_B(X,Y)
        return self.m, self.b

    def predict(self,X):
        y_pred = self.m * X + self.b
        return y_pred
