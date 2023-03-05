from math import log, ceil
from numpy import dot, zeros
from numpy.random import laplace
from SetsManager import get_sample

class PVarReducedFW:
    
    #https://arxiv.org/abs/2103.01516
    #Algorithm 3 but optimized for l1 ball
    
    def __init__(self, Z, Y, epsilon, f, number_of_phases):  
        
        self.S_Z = Z
        self.dim = len(self.S_Z[0])
        self.S_Y = Y
        self.epsilon = epsilon
        self.f = f
        x_start = zeros(self.dim)
        self.x_final = x_start
        self.T = number_of_phases
        self.pending = False

    def run(self):

        self.n = len(self.S_Z)
        self.m = self.dim
        self.b = self.n/(log(self.n)**2)

        for t in range(1, self.T + 1):
            self.t = t
            self.x = zeros((2**(t+1) + 1, self.dim))
            self.v = zeros((2**(t+1) + 1, self.dim))
            self.x[1] = self.x_final
            sampleS_Z, sampleS_Y = get_sample(self.S_Z, self.S_Y, int(self.b))
            self.v[1] = self.f.set_grad(self.x[1], sampleS_Y, sampleS_Z)
            self.DFS(0, 1, 1, 0)
            self.DFS(1, 1, 1, 1)
            self.x_final = self.x[2**t - 1]

        return

    def DFS(self, c, padre, j, binary):

        if self.pending == True:
            self.x[2*padre + c] = self.vector_pending
            self.pending = False

        if c == 0:
            self.x[2*padre] = self.x[padre]
            self.v[2*padre] = self.v[padre]
        else:
            sampleS_Z, sampleS_Y = get_sample(self.S_Z, self.S_Y, int(ceil(self.b//(2**j))))
            self.v[2*padre + 1] = self.v[padre] + self.f.set_grad(self.x[2*padre + 1], sampleS_Y, sampleS_Z) - self.f.set_grad(self.x[padre], sampleS_Y, sampleS_Z)
        if j == self.t:
            signo, coordenada = self.argmin(self.v[2*padre + c], (2*self.f.L0*2**self.t)/(self.b*self.epsilon))
            eta = 2/(2**(self.t-1) + binary + 1)
            self.vector_pending = (1-eta)*self.x[2*padre + c]
            self.vector_pending[coordenada] += eta*signo
            self.pending = True
        else:
            self.DFS(0, 2*padre+c, j+1, binary)
            self.DFS(1, 2*padre+c, j+1, binary + 2**(j))
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