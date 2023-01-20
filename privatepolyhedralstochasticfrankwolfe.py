from math import log, sqrt
from random import sample
from numpy import zeros, array, dot
from numpy.random import laplace
from loss import LossFunction
from sets_manager import partition

class PolySFW:
    
    #http://proceedings.mlr.press/v134/bassily21a/bassily21a.pdf
    #Algorithm 1
    
    def __init__(self, Z, Y, epsilon, delta, V, f, diametro, x_start): #recieve the input    
        self.S_Z = Z #private dataset data
        self.S_Y = Y #private dataset class
        self.epsilon = epsilon #privacy parameter
        self.delta = delta #privacy parameter
        self.V = V #set of vertices in the polyhedron
        self.f = f #loss function
        self.M = diametro
        self.x_start = x_start
    
    def run(self): #runs the algorithm

        self.n = len(self.S_Z) #dataset size
        self.K = len(self.V) #number of vertices
        self.dim = len(self.S_Z[0]) + 1 #dimension
        self.eta = log((self.n/log(self.K)))/self.n #step size

        if self.dim != len(self.V[0]):
            print("Las dimensiones no coinciden")
            return
        
        self.x = zeros((self.n//2+2, self.dim)) #x[i] is x^i
        self.d = zeros((self.n//2+1, self.dim)) #d[i] is d_i
        self.v = zeros(self.n//2+1) #v[i] is v_i
        self.s = zeros(self.n//2+1) #s[i] is s_i
        self.x[0] = self.x_start
        self.B_Z, self.B_Y, self.S_hat_Z, self.S_hat_Y = partition(self.S_Z, self.S_Y, (self.n+1)//2) #choose the partition
        
        self.d[0] = self.f.set_grad(self.x[0], self.B_Y, self.B_Z) #compute d_0
        self.v[0] = int(self.argmin(self.d[0], (4*self.f.L0*self.M*sqrt(log(1/self.delta)))/(self.epsilon*sqrt(self.n))))
        self.x[1] = (1-self.eta)*self.x[0]
        self.x[1] += self.eta*self.V[int(self.v[0])]
        for t in range(1, self.n//2):
            #print(f"el error en la iteración {t} con {self.x[t]} es {self.f.set_f(self.x[t], self.S_Y, self.S_Z)}")
            self.s[t] = max(((1-self.eta)**t)*((2*self.f.L0*self.M)/self.n), 2*self.eta*(self.f.L1*self.M**2 + self.f.L0*self.M))
            grad_var = self.f.grad(self.x[t], self.S_hat_Y[t-1], self.S_hat_Z[t-1]) - self.f.grad(self.x[t-1], self.S_hat_Y[t-1], self.S_hat_Z[t-1])
            self.d[t] = (1-self.eta)*(self.d[t-1] + grad_var) + self.eta*self.f.grad(self.x[t], self.S_hat_Y[t-1], self.S_hat_Z[t-1])
            self.v[t] = self.argmin(self.d[t], (2*self.s[t]*sqrt(self.n*log(1/self.delta)))/self.epsilon)
            self.x[t+1] = (1-self.eta)*self.x[t]
            self.x[t+1] += self.eta*self.V[int(self.v[t])]
        self.x_priv = self.x[self.n//2]
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