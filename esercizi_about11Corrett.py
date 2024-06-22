
from math import sqrt


class stats :
    '''calculator for statistics of a LIST of numbers'''

    summ = 0.
    sumSq = 0.
    N = 0
    sample = []

    def __init__ (self, sample):
        '''
        reads as input the collection of events,
        which needs to be a list of numbers
        '''
        self.sample = sample
        self.summ = sum (self.sample)
        self.sumSq = sum ([x*x for x in self.sample])
        self.N = len (self.sample)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def mean (self) :
        '''
        calculates the mean of the sample present in the object
        '''
        return self.summ / self.N

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def variance (self, bessel = True) :
        '''
        calculates the variance of the sample present in the object
        '''
        var = self.sumSq / self.N - self.mean () * self.mean ()
        if bessel : var = self.N * var / (self.N - 1)
        return var

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def sigma (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel))

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def sigma_mean (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel) / self.N)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def append (self, x):
        '''
        add an element to the sample
        '''
        self.sample.append (x)
        self.summ = self.summ + x
        self.sumSq = self.sumSq + x * x
        self.N = self.N + 1

#!/usr/bin/python

import random
from math import sqrt


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
# the model to be used for the exercise

def func (x, m, q) :
    '''
    reference model to be fitted
    '''
    return m * x + q

# initial parameters of the problem

m_true = 0.5
q_true = 1.1
epsilon_sigma = 0.3



###############################################
from math import floor
from matplotlib import pyplot as plt
import numpy as np


N_toys = 10000                                      #genero 10k toy esperiment che mi daranno 10k valori di Q2 sui rispettivi fit
Q_squares = []
x_coord_toy = np.arange (0, 10, 1)
y_coord_toy = np.zeros (10)
sigma_y = epsilon_sigma * np.ones (len (y_coord_toy))    
 
from iminuit import Minuit                           #importo la libreria iminuit
from iminuit.cost import LeastSquares

# generate a least-squares cost function  
least_squares = LeastSquares (x_coord_toy, y_coord_toy, sigma_y, func)             #1: blocco iminuit Q2, sarà presente anche nelle iterazioni
my_minuit = Minuit (least_squares, m = 0, q = 0)  # starting values for m and q
my_minuit.migrad ()  # finds minimum of least_squares function
my_minuit.hesse ()   # accurately computes uncertainties
# NB: adding additional instructions prevents the automatic visualisation of the fit result
#Q_squared = my_minuit.fval
#print ('value of the fit Q-squared', Q_squared)
N_dof = my_minuit.ndof
#print ('value of the number of degrees of freedom', N_dof)

for i_toy in range (N_toys) :
    epsilons_toy = generate_TCL_ms (0., epsilon_sigma, 10)             #numero aleatorio incertezza
    for i in range (x_coord_toy.size) :
        y_coord_toy[i] = func (x_coord_toy[i], m_true, q_true) + epsilons_toy[i]
    least_squares = LeastSquares (x_coord_toy, y_coord_toy, sigma_y, func)
    my_minuit_toy = Minuit (least_squares, m = 0, q = 0)  # starting values for m and q
    my_minuit_toy.migrad ()  # finds minimum of least_squares function
    my_minuit_toy.hesse ()   # accurately computes uncertainties
    if my_minuit_toy.valid : 
        Q_squares.append (my_minuit_toy.fval)

fig, ax = plt.subplots ()                                 #2: possibile blocco grafici
ax.set_title ('Q-squared distribution', size=14)
ax.set_xlabel('q_squared')
ax.set_ylabel('events in bin')
bin_edges = np.linspace (0, 4 * N_dof, floor (N_toys/100))   # edges o the histogram bins
ax.hist (Q_squares,
         bins = bin_edges,
         color = 'orange',
        )
plt.show ()

Q_squares_stats = stats (Q_squares)               #classe stats su Q_squares
print ('average Q_squared expected value:', Q_squares_stats.mean ())  #media sull'array Q_squares




Q_squares_stats = stats (Q_squares)                     #3: altre operazioni: teoria minimi quadrati e gdl
print ('average Q_squared expected value:', Q_squares_stats.mean ())
print ('sigma scale factor:', sqrt (Q_squares_stats.mean () / N_dof))
print ('sigma:', epsilon_sigma / sqrt (Q_squares_stats.mean () / N_dof))



# show the likelihood scans for the various parameters in 1D and 2D
my_minuit.draw_mnmatrix() #?