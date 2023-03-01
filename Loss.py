from math import exp, log
from numpy import dot, array, zeros

class LossFunction: #Now is implemented logistic loss (it can be changed)

    def __init__(self):
        self.L0 = 1 #f is L0-Lipschitz
        self.L1 = 1 #f is L1-smooth

    def sigmoid(self, z):
        return 1/(1+exp(-z))

    def f(self, x, y, z):
        # x is the actual parameter of the model
        # y is the real class (0 or 1)
        # z is the data (notice that dim(z) = dim(x) - 1)
        y_pred = self.sigmoid(dot(x, z))
        if y_pred == 1:
            return -1*(y*log(y_pred))
        res = -1*(y*log(y_pred) + (1-y)*log(1-y_pred))
        return res

    def set_f(self, x, Y, Z):
        # just f but Z, Y are set's of the data
        n = len(Z)
        res = sum(self.f(x, Y[i], Z[i]) for i in range(n))
        res = res/n
        return res

    def grad(self, x, y, z):
        # x is the actual parameter of the model
        # y is the real class (0 or 1)
        # z is the data (notice that dim(z) = dim(x) - 1)
        y_pred = self.sigmoid(dot(x, z))
        diff = y_pred - y
        x_grad = array([diff*z[i] for i in range(len(x))])
        return x_grad

    def set_grad(self, x, Y, Z):
        # just grad but Z, Y are set's of the data
        n = len(Z)
        m = len(x)
        res = zeros(m)
        for i in range(n):
            res = res + self.grad(x, Y[i], Z[i])
        if n == 0:
            return res
        return (1/n)*res