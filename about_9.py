
#1: generazione di numreri pc secondo una pdf expon e grafico

'''
Si scriva un programma che generi numeri pseudo-casuali distribuiti secondo una funzione esponenziale
Si faccia in modo che il tempo caratteristico dell’esponenziale ed numero di eventi da generare sia passato a linea di comando 
come parametro del programma eseguibile.
Riempire un istogramma con i numeri presenti 
dove sono stati trasferiti e che visualizzi l’istogramma a schermo.

'''

import numpy as np
import matplotlib.pyplot as plt
#from myrand import generate_exp
import random


def inv_exp(y)
def rand_exp (tau)

def generate_exp (tau, N, seed = 0.)


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



#2:
'''
Si scriva un programma che disegni in un GRAFICO la distribuzione di probabilità esponenziale avendo fissato il parametro t0 
ad un numero passato da linea di comando come parametro del programma eseguibile.

'''
#from likelihood import exp_pdf
def exp_pdf (x, tau) :
    '''
    the exponential probability density function
    '''
    if tau == 0. : return 1.
    return exp (-1 * x / tau) / tau

#inputs:
xMin=0
xMax=8
tau= 1

fig, ax = plt.subplots ()
ax.set_title ('Exponential distribution', size=14)
ax.set_xlabel ('time')
ax.set_ylabel ('probability density')

x = np.linspace (xMin, xMax, 10000)
y = np.arange (0., x.size)
for i in range (x.size):
  y[i] = exp_pdf (x[i], tau)  #<--
plt.plot (x, y, 'g')
plt.show()


#3: FUNZIONE LIKELIHOOD (variando i t0) su distr exp
#fare lo stesso sostiutendo con la LOGLIKELIHOOD
'''
Si scriva una funzione loglikelihood che calcoli il logaritmo della verosimiglianza al variare del parametro t0,
per un campione di eventi pseudo-casuali generati secondo le istruzioni dell’Esercizio 1. 
Si scriva la funzione in modo che possa essere utilizzata per costruire un grafico che ne disegni l’andamento in funzione di t0
'''

from math import exp
import matplotlib.pyplot as plt
import numpy as np
import random

def exp_pdf (x, tau) :
def loglikelihood (theta, pdf, sample) :
   

#inputs:
xMin=0
xMax=8
tau= 1

x = np.linspace (xMin, xMax, 10000)
sample = []
for i in range (x.size):
  sample.append(exp_pdf (x[i], tau))

#print(likelihood (tau, exp_pdf, sample))

#VARIAMO IL TAU 

taus = np.linspace(0.1,1,500)
xx= np.random.uniform(0,1,10)
sample2 = []
for i in range(0,taus.size):
    t0=taus[i]
    sample2.append(likelihood(t0,exp_pdf,xx))

plt.plot(taus,sample2)
plt.show()


#3.b
# reducing by a factor 10 the initial sample
subsample = sample[:int (len (sample)/10)]

fig, ax = plt.subplots ()
ax.set_title ('Log-likelihood scan', size=14)
ax.set_xlabel ('tau')
ax.set_ylabel ('log likelihood')

ll_1 = np.arange (0., tau.size)
for i in range (tau.size) :
    ll_1[i] = loglikelihood (tau[i], exp_pdf, subsample)

plt.plot (tau, ll_1, 'r')
plt.show ()   


