from math import log, ceil
from numpy import dot, zeros
from numpy.random import laplace
from sets_manager import get_sample

class PVarReducedFW:
    
    #https://arxiv.org/abs/2103.01516
    #Algorithm 3
    
    def __init__(self, Z, Y, epsilon, V, f, diametro, x_start, number_of_phases): #recieve the input    
        
        self.S_Z = Z #private dataset data
        self.S_Y = Y #private dataset class
        self.epsilon = epsilon #privacy parameter
        self.V = V #set of vertices in the polyhedron
        self.f = f #loss function
        self.x_final = x_start #initial point
        self.T = number_of_phases #number of phases
        self.pending = False
        self.D = diametro

    def run(self): #runs the algorithm

        self.n = len(self.S_Z) #dataset size
        self.m = len(self.V) #number of vertices
        self.dim = len(self.S_Z[0]) + 1 #dimension
        self.b = self.n/(log(self.n)**2) #batch size

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
            w = self.argmin(self.v[2*padre + c], (2*self.f.L0*self.D*2**self.t)/(self.b*self.epsilon))
            eta = 2/(2**(self.t-1) + binary + 1)
            self.vector_pending = (1-eta)*self.x[2*padre + c] + eta*self.V[int(w)]
            self.pending = True
        else:
            self.DFS(0, 2*padre+c, j+1, binary)
            self.DFS(1, 2*padre+c, j+1, binary + 2**(j))
        return

    def argmin(self, d, beta):
        mini = dot(self.V[0], d) + laplace(beta)
        where = 0
        for i in range(1, len(self.V)):
            value = dot(self.V[i], d) + laplace(beta)
            if value < mini:
                mini = value
                where = i
        return where