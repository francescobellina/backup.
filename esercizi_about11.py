

#esercizio 1

#scrivere in una libreria dedicata una funzione lineare phi(x,teta) con
# 2 parametri teta ( teta0=(teta1,teta2) )
#- scrivi un programma che generi un set di 10 paia (xi,yi) in modo tale
#che i punti xi siano distribuiti in maniera pseudo-causale lungo l'asse
#orizzontale tra i valori 0 - 10; mentre i punti yi derivano dalla 
#formula: yi= phi(xi,teta) + ei
#-plotta il tutto includendo le barre di errore


from math import sqrt
import random

def generate_uniform (N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra 0 ed 1
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (random.random ())
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_range (xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra xMin ed xMax
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_range (xMin, xMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TAC (f, xMin, xMax, yMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TAC (f, xMin, xMax, yMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TAC (f, xMin, xMax, yMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL (xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL_ms (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


#--------------------------------------------------------
#--------------------------------------------------------

from matplotlib import pyplot as plt 
import numpy as np

def phi (x, m, q) :
  """a linear function

  Args:
      x (float): the `x` value
      m (float): the slope
      q (float): the intercept
  """    
  return  x*m + q
'''
m_ = 0.5
q_ = 1.1
epsilon_sigma = 0.3


x_values = generate_range (0, 1, 10, seed = 0.) 
print(x_values) 
x_values2 = np.arange (0, 10,1) #SONO EQUISTANZIATI la aleatorietà è data da y con phi e epsilon
print(x_values) 

epsilons = generate_TCL_ms (0., epsilon_sigma, 10)

y_values = np.zeros(10)
for i in range(len(x_values2)): y_values[i]= (phi(x_values2[i], m_, q_) + epsilons[i])


sigma_y = np.ones(10)*epsilon_sigma 
'''

#plotting:
'''
fig, ax = plt.subplots()
ax.set_title('modello lineare', size=14)
ax.set_xlabel ('x')
ax.set_ylabel ('y')
ax.errorbar(x_values2, y_values, xerr = 0.0, yerr = sigma_y, linestyle = 'None', marker = 'o')
plt.show()
'''

#PARTE2:
#Use the iMinuit library to perform a fit on the simulated sample.
#guarda se il fit converge con successo
#stampa i valori dei parametri determinati (su m e q) e le loro
#deviazioni rispettive standard (sigmas)

from iminuit import Minuit
from iminuit.cost import LeastSquares
'''

least_squares = LeastSquares (x_values2, y_values, sigma_y, phi)  #Q^2
my_minuit = Minuit (least_squares, m = 0, q = 0)
my_minuit.migrad ()  # finds minimum of least_squares function
my_minuit.hesse ()

is_valid = my_minuit.valid
print ('success of the fit: ', is_valid)

for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
    print(f'{par} = {val:.3f} +/- {err:.3f}') 

#print (my_minuit.covariance)

m_fit = my_minuit.values[0]
q_fit = my_minuit.values[1]  
#display (my_minuit)


'''

#PARTE3:
#-Calcola il Q2 usando i punti del sample e fittando la funzione phi
#-compara il valore ottenuto con iminuit con quello calcolato "a mano"
#-stampa il valore del numero di gradi di libertà del fit
'''
least_squares = LeastSquares (x_values2, y_values, sigma_y, phi)
Q_squared = my_minuit.fval
print ('value of the fit Q-squared', Q_squared)

N_dof = my_minuit.ndof
print ('value of the number of degrees of freedom', N_dof)

#ora procediamo al calcolo del Q2 da confrontare con il risultato ottenuto con iminuit:

Q_value = 0

for i in range(len(y_values)): Q_value = Q_value + pow (((y_values[i]- (phi(x_values2[i], m_fit, q_fit)))/sigma_y[i]) , 2)

print('valore del Q-squared: ', Q_value)

Q_squared_calc = 0.
for x, y, ey in zip (x_values2, y_values, sigma_y) :
    Q_squared_calc = Q_squared_calc + pow ( (y - func (x, m_fit, q_fit)) /ey , 2 )  
print ('valore del Q-squared: ' , Q_squared_calc) 

'''

#PARTE 4:
#Using the toy experiments technique, generate 10,000 fit experiments with the model studied 
#in the previous exercises and fill a histogram with the obtained values of Q2
#Compare the expected value of Q2 obtained from the toy experiments with the degrees of freedom of the problem.
from math import floor
#import stats

N_toy = 10000   
Q_due = []

x_values2 = np.arange (0, 10,1)
epsilon_sigma = 0.3
m_ = 0.5
q_ = 1.1
sigma_y = np.ones(10)*epsilon_sigma 

# loop over toys
for i in range (N_toy):
    y_values = np.zeros(10)
    epsilons = generate_TCL_ms (0., epsilon_sigma, 10, 10, i)
    for j in range(len(x_values2)): y_values[j]= (phi(x_values2[j], m_, q_) + epsilons[j])
    least_squares = LeastSquares (x_values2, y_values, sigma_y, phi)  #Q^2
    my_minuit = Minuit (least_squares, m = 0, q = 0)
    my_minuit.migrad ()  # finds minimum of least_squares function
    my_minuit.hesse ()
    least_squares = LeastSquares (x_values2, y_values, sigma_y, phi)
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof
    if my_minuit.valid : 
        Q_due.append (my_minuit.fval)
    
#print (Q_due)



fig, ax = plt.subplots ()
ax.set_title ('Q-squared distribution', size=14)
ax.set_xlabel('q_squared')
ax.set_ylabel('events in bin')
bin_edges = np.linspace (0.4 * N_dof, floor (N_toy/100))   # edges o the histogram bins
ax.hist (Q_due,
         bins = bin_edges,
         color = 'orange',
        )
plt.show ()

#Q_due_stats = stats (Q_due)
#print ('average Q_squared expected value:', Q_due_stats.mean ())