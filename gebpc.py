
#Write a Python program to draw a Gaussian distribution and its cumulative function
#Write a Python program to draw an exponential distribution and its cumulative function
#Write a Python program to draw a binomial distribution and its cumulative function
#Write a Python program to draw a Poisson distribution for several values of its mean, overlapped

from scipy.stats import norm
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from scipy.stats import norm, expon, binom, poisson
#--
def Gaussian (x, mean, sigma) :   #FUNZIONE GAUSSIANA
    if sigma == 0 : return float (x == 0.)
    return np.exp (-0.5 * pow ((x - mean) / sigma, 2.)) / (sqrt (2 * np.pi) * sigma) #forma analitica

#-
mean = 1.
sigma = 0.5
x = mean + sigma / 2.
norm.pdf (x, mean, sigma)

#--
#1:NORMALE
normal = norm (100., 10.) #I DUE VALORI SONO LA MEDIA E LA SIGMA (che di default sono 0 e 1)

x_axis = np.linspace (50., 150., 100) #input sulle x

plt.plot(x_axis, normal.pdf (x_axis), label="PDF")
plt.legend ()
#plt.savefig ('ex_3.7_pdf.png')
plt.show()

plt.clf () # clear the figure COSI ME NE PLOTTA DUE CONSECUTIVE

plt.plot (x_axis, normal.cdf (x_axis), color="red",label="CDF")
plt.legend ()
plt.show()
#plt.savefig ('ex_3.7_cdf.png')

#OPPURE DI PER PLOTTARE prima sovrapposti e poi DIFIANCO:
def main () :
    '''
    Funzione che implementa il programma principale
    '''
    normal = norm (100., 10.)
    x_axis = np.linspace (50., 150., 100)
    plt.plot (x_axis, normal.pdf (x_axis)*10, label="PDF")
    plt.plot (x_axis, normal.cdf (x_axis), label="CDF")
    plt.legend ()
    #plt.show()

#OPPURE affiancati:
    fig, ax = plt.subplots(1,2)
    ax[0].plot (x_axis, normal.pdf (x_axis), label="PDF", color= 'red')
    ax[0].legend()
    ax[1].plot (x_axis, normal.cdf (x_axis), label="CDF")
    ax[1].legend()
    plt.show()
    

#--
#2:ESPONENZIALE
import matplotlib.pyplot as plt
from scipy.stats import expon
import numpy as np

def main () :
    '''
    Funzione che implementa il programma principale
    '''

    tau = 2.
    expo = expon (0., tau) #DISTRIBUZIONE ESPONENZIALE
    x_axis = np.linspace (0., 8., 100)


    fig, ax = plt.subplots (nrows = 1, ncols = 1)

    ax.plot (x_axis, expo.pdf (x_axis), label="PDF")
    ax.plot (x_axis, expo.cdf (x_axis), label='CDF')
    ax.set_title ('Comparing exponenzial distr. with its cdf', size=14)
    ax.legend ()
    plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


if __name__ == "__main__":
    main ()



#--
#3:BINOMIALE

import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    success_p = 0.5
    N = 8
    binomial = binom (N, success_p)
    x_axis = np.arange (N+1)


    fig, ax = plt.subplots (1,1)

    ax.scatter(x_axis, binomial.pmf (x_axis), label='PMF')#grafico scatterato, PMF: probability mass function (funzione di prob, densità discreta)
    ax.scatter (x_axis, binomial.cdf (x_axis), label='CDF')
    plt.legend ()
    plt.show()
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


if __name__ == "__main__":
    main ()

#4:POISSONIANA
#!/usr/bin/python
'''
Write a Python program to draw a Poisson distribution and its cumulative function
'''

import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    average = 4.
    poiss = poisson (average)        #implementazione e richiamo della funzione poissoniana (distribuzione)
    x_axis = np.arange (3 * average)
    
    fig,ax = plt.subplots(1,1)
    ax.scatter (x_axis, poiss.pmf (x_axis), label='PMF')
    ax.scatter (x_axis, poiss.cdf (x_axis), label='CDF')

    plt.title ("Poisson Distribution")
    plt.ylabel ("Density")
    plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


if __name__ == "__main__":
    main ()


#4.b: SKEWNESS E KURTOSI DI UNA POISSONIANA
#!/usr/bin/python
'''
Write a Python program to draw a Poisson distribution
Show, by using the third and fourth central momenta calculations available in the `scipy.stat` library,
that the momenta of a Poisson distribution asymptotically tend to the ones of a Gaussian
cioè tendono a zero (come per la guassiana che è fissa a zero) all'aumentare delle misure
'''

import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    skew_l = [] #SKEWNESS 
    kurt_l = [] #KURTOSIS
    ave_l = np.arange(0,100)

    for average in ave_l :
        poiss = poisson (average)
        ave, var, skew, kurt = poiss.stats (moments='mvsk') #cosi richiamo i vari momenti
        skew_l.append (skew) #al variare della media aggiungo i valori
        kurt_l.append (kurt)
    
    fig,ax = plt.subplots(1,1)
    ax.scatter (ave_l, skew_l, label='skewness',s=4)
    ax.scatter (ave_l, kurt_l, label='kurtosis', s=4)

    plt.title ('Poisson skewness and kurtosis')
    plt.xlabel ('mean')
    plt.xlabel ('skewness and kurtosis vs mean')
    plt.legend()
    plt.show()



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()




#----------------------------------------------------------
#----------------------------------------------------------

#ENCORE:

#1:
#CALCOLO AREA CODE DELLA DISTR GAUSSIANA con scipy.stat.norm da sigma 1 a sigma 5.
'''
Use the Python `scipy.stat.norm` object to determine the area of a normal distribution
of its tails outside the range included within an interval of 1, 2, 3, 4, and 5
standard deviations around its mean
'''

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def main () :
    '''
    Funzione che implementa il programma principale
    '''
    normal = norm (0., 1.) #ma possono variari i parametri
    for sigmas in range (1,6) : #da 1-sigma a 5-sigma
        tails_area = normal.cdf (0.- sigmas) + 1. - normal.cdf (0.+ sigmas) #QUESTA è la formula

        print ('outside ' + str (sigmas) + ' sigmas :\t'
               + str (tails_area))

#Al crescere delle sigma diminuisce l'area


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

if __name__ == "__main__":
    main ()



#2:
#probabilità che un numero ripetuto (10) di misure di un certo EVENTO GAUSSIANO (e poi POSSIONIANO)
#cadano entro una deviazione standard attorno al valor medio

'''
What is the probability that 10 measurements of the same quantity
expected to be Gaussian(poissonian) fall within an interval of 1 standard deviation width 
around the mean?
'''

from scipy.stats import norm, poisson
import numpy as np

def main () :
    '''
    Funzione che implementa il programma principale
    '''
    
    mean = 0.   #sono quelli di default ma possono ovviamente cambiare
    sigma = 1.
    normal = norm (mean, sigma)
    x_min = mean - sigma*0.5  #metà sigma
    x_max = mean + sigma*0.5
    single_evt_prob = normal.cdf (x_max) - normal.cdf (x_min) #calcolo mediante la cdf
    print ('single event probability: ', single_evt_prob)
    print ('joint probability: ', single_evt_prob**10)

#PASSIAMo ora al caso di evento possioniano

    average = 4.
    poiss = poisson (average)
    single_evt_prob = 1. - poiss.cdf (average)
    print ('single event probability: ', single_evt_prob)
    print ('joint probability: ', single_evt_prob**10)





# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


if __name__ == "__main__":
    main ()



#3: generazione di numeri pc secondo una pdf ESPONENZIALE e istogramma

import numpy as np
import matplotlib.pyplot as plt
#from myrand import generate_exp
import random


def inv_exp (y, lamb = 1) :
    '''
    Inverse of the primitive of the exponential PDF.
    pdf(x) = lambda * exp(-lambda x) x >= 0, 0 otherwise.
    F(x) = int_{0}^{x} pdf(x)dx = 1 - exp(-lambda * x) for x >= 0, 0 otherwise.
    F^{-1}(y) = - (ln(1-y)) / lambda
    '''
    return -1 * np.log (1-y) / lamb



def rand_exp (tau) :
    '''
    generazione di un numero pseudo-casuale esponenziale
    con il metodo della funzione inversa
    a partire dal tau dell'esponenziale
    '''
    lamb = 1. / tau
    return inv_exp (random.random (), lamb)


def generate_exp (tau, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali esponenziali
    con il metodo della funzione inversa, noto tau dell'esponenziale,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_exp (tau))
    return randlist


tau   = 2.
N_evt = 800

sample = generate_exp (tau, N_evt)

xMin = 0.                                    # minimum of the histogram drawing range
xMax = 4. * tau                              # maximum of the histogram drawing range
bin_edges = np.arange (xMin, xMax, tau/4.)   # edges o the histogram bins

# disegno della funzione
fig, ax = plt.subplots ()
ax.set_title ('Exponential sample', size=14)
ax.set_xlabel('time')
ax.set_ylabel('events in bin')
ax.hist (sample,
         bins = bin_edges,
         color = 'orange',
        )
plt.show ()