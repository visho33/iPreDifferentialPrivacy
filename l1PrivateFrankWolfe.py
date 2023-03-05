from math import log, sqrt
from random import sample
from numpy import zeros, array, dot
from numpy.random import laplace
from Loss import LossFunction
from SetsManager import partition

class PolySFW:
    
    #http://proceedings.mlr.press/v134/bassily21a/bassily21a.pdf
    #Algorithm 1 optimized for l1 ball
    
    def __init__(self, Z, Y, epsilon, delta, f):    
        self.S_Z = Z
        self.S_Y = Y
        self.epsilon = epsilon
        self.delta = delta
        self.f = f
    
    def run(self):

        self.n = len(self.S_Z)
        self.dim = len(self.S_Z[0])
        self.K = self.dim
        self.eta = log((self.n/log(self.K)))/self.n
        
        self.B_Z, self.B_Y, self.S_hat_Z, self.S_hat_Y = partition(self.S_Z, self.S_Y, (self.n+1)//2)
        
        
        self.x_old = zeros(self.dim)
        self.d = self.f.set_grad(self.x_old, self.B_Y, self.B_Z)
        signo, coordenada = self.argmin(self.d, (4*self.f.L0*2*sqrt(log(1/self.delta)))/(self.epsilon*sqrt(self.n)))
        self.x = (1-self.eta)*self.x_old
        self.x[coordenada] += self.eta*signo


        for t in range(1, self.n//2):
            self.s = max(((1-self.eta)**t)*((2*self.f.L0*2)/self.n), 2*self.eta*(self.f.L1*4 + self.f.L0*2))
            grad_var = self.f.grad(self.x, self.S_hat_Y[t-1], self.S_hat_Z[t-1]) - self.f.grad(self.x_old, self.S_hat_Y[t-1], self.S_hat_Z[t-1])
            self.d = (1-self.eta)*(self.d + grad_var) + self.eta*self.f.grad(self.x, self.S_hat_Y[t-1], self.S_hat_Z[t-1])
            signo, coordenada = self.argmin(self.d, (2*self.s*sqrt(self.n*log(1/self.delta)))/self.epsilon)
            self.x_old = self.x
            self.x = (1-self.eta)*self.x_old
            self.x[coordenada] += self.eta*signo
        
        self.x_priv = self.x
        return

    def argmin(self, d, beta):
        mini = d[0] + laplace(beta)
        signo_mini = 1
        coordenada_mini = 0
        for coordenada in range(self.dim):
            for signo in [1, -1]:
                value = signo*d[coordenada] + laplace(beta)
                if value < mini:
                    mini = value
                    signo_mini = signo
                    coordenada_mini = coordenada
        return signo_mini, coordenada_mini