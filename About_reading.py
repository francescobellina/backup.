# load the data table. IMPORTARE UN FILE DI TESTO !
import numpy as np


redshift, distanza, sigma = np.loadtxt('\\Users\Francesco\Desktop\SuperNovae.txt', unpack=True)

print(distanza[:10])
print(len(redshift))
print(sigma.dtype)



#---------------------
#FUNZIONE IMPLICITA LAMBDA
#definizione 1:
print ((lambda x : x**2)(number))
#number è la variabile su cui faccio agire la funzione, x è di supporto (posso usare anche un altra lettera)

#definizione 2:
func = lambda x : x**2
print (func (number))


#MAP e FILTER

lista = list (range (-5, 5))

squared = list (map (lambda x : x**2, lista))
print(squared)

select = list (filter (lambda x: x!=0, range(-5, 5)))
print(select)

#ZIP

lista1= list(range(-5,5))
lista2 = list (map (lambda x : x**2, lista1))

for e, esq in zip (lista1, lista2) :
    print (e, esq)

#ZIP2
import numpy as np
from math import sin

lista1= list(range(-5,5))
lista2 = list (map (lambda x : x**2, lista1))
lista3 = list(range(0,10))
lista4 = list(map(lambda x: x*(-1),lista1))
lista4= list(np.arange(0,10))

for a, b, c, d, e in zip (lista1, lista4, lista2, lista3, lista4):
    print(a, b, c, d, e)




#SYS:

import sys 

    if len(sys.argv) < 6 :
        print ('usage: ', sys.argv[0], 'xMin xMax yMin yMax N_evt_min N_evt_max')
        exit ()
    

    xMin = float (sys.argv[1])
    if ...
    xMax = float (sys.argv[2])
    yMax = float (sys.argv[3])
    N_evt_min = int (sys.argv[4])
    N_evt_max = int (sys.argv[5])