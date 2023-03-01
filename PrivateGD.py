from math import sqrt, log
from numpy import zeros
from numpy.random import normal

class NoisyGD:
    
    #https://arxiv.org/abs/2205.03014
    #Algorithm 1
    
    def __init__(self, Z, Y, f, B, epsilon, delta): #recieve the input    
        
        self.Z = Z #dataset
        self.Y = Y #dataset
        self.f = f #loss function
        self.B = B #l2 diameter
        self.n = len(Z)
        self.T = self.n #number of steps
        G = 2*sqrt(self.f.L1) + 2*f.L1*B
        self.sigma2 = (8*(G**2)*self.T*log(1/delta))/((self.n**2)*(epsilon**2)) #noise scale
        self.eta = min(B/(sqrt(self.T)*max(sqrt(f.L1), sqrt(self.sigma2*(self.n*epsilon)**(2/3)))), 1/(4*f.L1)) #learning rate
    
    def run(self): #runs the algorithm

        self.dim = len(self.Z[0])
        self.x = zeros((self.T+1, self.dim))
        for t in range(self.T):
            self.x[t+1] = self.x[t] - self.eta*(self.gaussian_mech(self.f.grad(self.x[t], self.Y[t], self.Z[t])))
            self.x[t+1] = self.normalize(self.x[t+1])

        self.x_final = self.x[self.T]
        return

    def gaussian_mech(self, vec):
        return vec + normal(loc = 0, scale = self.sigma2, size = len(vec))
    
    def normalize(self, vec):
        return self.B*(1/sqrt(sum(v**2 for v in vec)))*vec