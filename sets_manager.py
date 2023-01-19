from numpy import array, zeros
from random import sample
from numpy.random import choice

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