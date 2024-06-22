import numpy as np
import math
from random import uniform

#---------------

def parabol(x,a):
    return x**2 + a

#---------------
#Si generino N = 10 punti xi distribuiti in modo pseudo-casuale
#secondo una distribuzione uniforme sull'intervallo orizziontale e si associ a ciascuno di essi una coordinata
#TRE MODI DIFFERENTI (sintassi)

x_coord = np.array ([uniform (0, 10) for i in range (10)])
x_coord.sort () # i punti vanno ordinati perché iminuit faccia il fit nel range che li contiene tutti
print(x_coord)

x=[]
for i in range(10):
    x.append(uniform(0,10))
x.sort()
print(x)


z=np.zeros(10)
for i in range(10):
    z[i]=uniform(0,10)
z.sort()
print("le z valgono: ", z)



#e si associ a ciascuno di essi una coordinata yi dove εi è un numero pseudo casuale generato,
#con il metodo del teorema centrale del limite, secondo una distribuzione Gaussiana di media 0 e deviazione standard σy=10.

#la funzione è y=x**2 + e, con epsilon generato uniformemente (poi faremo con TLC_ms)
#quindi a ciascun xi devo associare una coordinata cosi trasformata.

sigma_y = 10
a=0

y=np.zeros(10)
for i in range(len(z)):
    y[i] = parabol(z[i],a) + uniform(0,1) 

print("le yi con lo scarto dato da uniform : ", y)

#y_coord = list( map (lambda k:sum(k), zip (phi (x_coord, a, b, c), generate_TCL_ms (0., sigma_y, 10))))
#y_coord = list( map (lambda k:sum(k), zip (parabol (z, a), uniform (0, 1))))