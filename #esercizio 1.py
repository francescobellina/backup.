#esercizio 1

#scrivere in una libreria dedicata una funzione lineare phi(x,teta) con
# 2 parametri teta ( teta0=(teta1,teta2) )
#- scrivi un programma che generi un set di 10 paia (xi,yi) in modo tale
#che i punti xi siano distribuiti in maniera pseudo-causale lungo l'asse
#orizzontale tra i valori 0 - 10; mentre i punti yi derivano dalla 
#forula: yi= phi(xi,teta) + ei
#-plotta il tutto includendo le barre di errore

import myrand
from myrand import generate_TCL_ms
from matplotlib import pyplot as plt
import matplotlib.pyplot.errorbar 
import numpy as np

def phi (x, m, q) :
  """a linear function

  Args:
      x (float): the `x` value
      m (float): the slope
      q (float): the intercept
  """    
  return  m*x + q

x_values = np.generate.uniform(0,10) 