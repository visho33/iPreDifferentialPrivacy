from numpy import array, zeros, dot
from random import sample
from numpy.random import choice, rand
from math import exp

def get_l1_ball(dim, radius):
    #get the vertices of l1 ball of a given radius
    final = zeros((2*dim, dim))
    for i in range(2*dim):
        final[i][i//2] = radius*(1-2*(i%2))
    return final

def partition(Z, Y, size):
    #partitionate the set into a subset of a given size, and the other the complement
    selected = choice(Z.shape[0], size, replace=False)
    selected.sort()
    non_selected = []
    j = 0
    for i in range(len(Z)):
        if j >= len(selected) or selected[j] != i:
            non_selected.append(i)
        else:
            j += 1
    non_selected = array(non_selected)
    non_selected.sort()
    return Z[selected], Y[selected], Z[non_selected], Y[non_selected]

def get_sample(Z, Y, size):
    selected = choice(Z.shape[0], size, replace=False)
    return Z[selected], Y[selected]

def sigmoid(z):
    return 1/(1+exp(-z))

def generate_data(x_star, size, max_cordinate, EPS):
    #create a dataset of given size using x_star, the probability of being incorrect class is EPS
    dim = len(x_star)
    
    Z = zeros((size, dim - 1))
    Y = zeros(size)
    largo = 0
    
    while largo < size:
        z = max_cordinate*rand(dim - 1) - max_cordinate
        pred = sigmoid(dot(x_star[1:], z) + x_star[0])
        if pred < EPS:
            Z[largo] = z
            Y[largo] = 0
            largo += 1
        if pred > 1 - EPS:
            Z[largo] = z
            Y[largo] = 1
            largo += 1
    
    return array(Y), array(Z)
