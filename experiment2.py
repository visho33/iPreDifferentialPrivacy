from pandas import read_csv
from numpy import array, zeros
from sets_manager import get_l1_ball
from loss import LossFunction
from privatevariancereducedfrankwolfe import PVarReducedFW

print("cargando data-frame...")
df = read_csv('data.csv')
print("guardando data-frame...")
Y = array(df["y"])
df.drop('id', inplace = True, axis = 1)
df.drop('y', inplace = True, axis = 1)
df_new = df.iloc[:, 0:2]
Z = array(df_new)
dim = len(Z[0]) + 1
radius = 1
print(f"obteniendo el conjunto de {dim} vertices...")
vertices = get_l1_ball(dim, radius)
f = LossFunction()
f.L0 = 8*radius
f.L1 = 2
epsilon = 2
x = zeros(dim)
print("creando el modelo...")
x_zero = array([(1-2*(i%2))/(dim) for i in range(dim)])
print(x_zero)
print("el error con el vector inicial es ", f.set_f(x_zero, Y, Z))
model2 = PVarReducedFW(Z, Y, epsilon, vertices, f, 2*radius, x_zero, 10)
print("corriendo el modelo...")
model2.run()
print("se obtuvo el siguiente punto óptimo", model2.x_final)
print(f.set_f(model2.x_final, Y, Z))