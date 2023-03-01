from math import sqrt, log
from numpy import zeros
from numpy.random import normal

class NoisyGD:
    
    #https://arxiv.org/abs/2205.03014
    #Algorithm 1 but not private
    
    def __init__(self, Z, Y, f, B): #recieve the input    
        
        self.Z = Z #dataset
        self.Y = Y #dataset
        self.f = f #loss function
        self.B = B #l2 diameter
        self.n = len(Z)
        self.T = self.n #number of steps
        G = 2*sqrt(self.f.L1) + 2*f.L1*B
        self.eta = 1/(4*f.L1) #learning rate
    
    def run(self): #runs the algorithm

        self.dim = len(self.Z[0])
        self.x = zeros((self.T+1, self.dim))
        for t in range(self.T):
            self.x[t+1] = self.x[t] - self.eta*self.f.grad(self.x[t], self.Y[t], self.Z[t])
            self.x[t+1] = self.normalize(self.x[t+1])
        self.x_final = self.x[self.T]
        return
    
    def normalize(self, vec):
        return self.B*(1/sqrt(sum(v**2 for v in vec)))*vec 
