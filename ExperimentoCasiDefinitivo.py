from numpy import zeros
from sets_manager import get_l1_ball, generate_data
from loss import LossFunction
from privatevariancereducedfrankwolfe import PVarReducedFW
from numpy.random import rand
from matplotlib.pyplot import hist, title, xlabel, ylabel, xticks, show

errores = []
intervalos = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

for t in range(100):
    print(t)
    dimx = 10
    radius = 1
    radius_data = 5*dimx*dimx
    x_star = 1/dimx*(radius*rand(dimx) - radius)
    Y, Z = generate_data(x_star, 50000, radius_data, 0.05)
    dim = len(Z[0]) + 1
    vertices = get_l1_ball(dim, radius)
    f = LossFunction()
    f.L0 = 8*radius
    f.L1 = 2
    epsilon = 1
    x_zero = zeros(dim)
    model1 = PVarReducedFW(Z, Y, epsilon, vertices, f, 2*radius, x_zero, 10)
    model1.run()
    errores.append(min(f.set_f(model1.x_final, Y, Z), 1.55))

hist(x = errores, bins = intervalos)
title("Errores Frank Wolfe Privado con Varianza Reducida")
xlabel("Error")
ylabel("Frecuencia")
xticks(intervalos)
show()