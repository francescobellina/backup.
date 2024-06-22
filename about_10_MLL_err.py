
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import random

from math import exp, log

def inv_exp (y, lamb = 1) :
def generate_exp (tau, N, seed = 0.) :
def exp_pdf (x, tau) :
  
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def loglikelihood (theta, pdf, sample) :
    
    
#inputs:
xMin=0
xMax=8
tau= 1

#print(likelihood (tau, exp_pdf, sample))

#VARIAMO IL TAU 

taus = np.linspace(0.1,1,500)
xx= np.random.uniform(0,1,10)
sample2 = []
for i in range(0,taus.size):
    t0=taus[i]
    sample2.append(loglikelihood(t0,exp_pdf,xx))

plt.plot(taus,sample2)
plt.show()



#1: TROVARE IL TAU (medio, stimatore) : trovo il massimo della logLL con il metodo golden ratio


def sezioneAureaMax_LL (
    g,              # funzione di likelihood trovare il massimo
    pdf,            # probability density function of the events
    sample,         # sample of the events
    x0,             # estremo dell'intervallo  -->(tau min)        
    x1,             # altro estremo dell'intervallo -->   (tau max)     
    prec = 0.0001): # precisione della funzione   


tau_true  = 2.
N_evt     = 50

sample = generate_exp (tau_true, N_evt) #per il confronto
tau_hat = sezioneAureaMax_LL (loglikelihood, exp_pdf, sample, 0.5, 5.)
print ('il valore di tau che massimizza il logaritmo della verosimiglianza è:', tau_hat)

#------
#2: plottare con il punto trovato
#Plot the profile of the likelihood function and the point identified as its maximum.

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots ()
ax.set_title ('Log-likelihood scan', size=14)
ax.set_xlabel ('tau')
ax.set_ylabel ('log likelihood')

tauaxis = np.linspace (0.5, 5., 10000)
ll      = np.arange (0., tauaxis.size)

for i in range (tauaxis.size) :
    ll[i] = loglikelihood (tauaxis[i], exp_pdf, sample)

plt.plot (tauaxis, ll, 'r')
plt.plot ([tau_hat, tau_hat],ax.get_ylim (), color = 'blue')
plt.text (tau_hat * 1.1, plt.ylim ()[0] + 0.1 * (ax.get_ylim ()[1] - ax.get_ylim ()[0]), 'tau_hat =' + str (tau_hat))
plt.show ()

#3:INCERTEZZA SULLO STIMATORE
#Lo stimatore di τ è una variabile casuale, cioè ha una propria distribuzione di probabilità
#Dunque oltre al avere associata una stima puntuale ricavata massimizzando il logaritmo della verosimiglianza possiede anche una sigma
#Si utilizza spesso un metodo grafico per determinare questa sigma, che si basa sul fatto
#  che asintoticamente la funzione di likelihood RISPETTO AI PARAMETRI è Gaussiana,
# dunque che la funzione di log-likelihood è parabolica : C'è UNA FUNzione di 
'''
Si utilizzi il metodo della bisezione per trovare i due punti τ - στ e τ + στ nel caso dell’esercizio precedente.
Si confronti il valore di στ ottenuto in questo modo con quello calcolato a partire dalla media aritmetica dei numeri salvati 
'''
'''
def intersect_LLR (
    g,              # funzione di cui trovare lo zero
    pdf,            # probability density function of the events
    sample,         # sample of the events
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo 
    ylevel,         # value of the horizontal intersection    
    theta_hat,      # maximum of the likelihood    
    prec = 0.0001): # precisione della funzione        

    def gprime (x) :
        return g (x, pdf, sample, theta_hat) - ylevel    <---------------------!!!!

    xAve = xMin 
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin) 
        if (gprime (xAve) * gprime (xMin) > 0.) : xMin = xAve 
        else                                    : xMax = xAve 
    return xAve 


tau_hat = sezioneAureaMax_LL (loglikelihood, exp_pdf, sample, 0.5, 5., 0.0001)    
tau_hat_minusS = intersect_LLR (loglikelihood, exp_pdf, sample, 0.5, tau_hat, -0.5, tau_hat)
tau_hat_plusS = intersect_LLR (loglikelihood, exp_pdf, sample, tau_hat, 5., -0.5, tau_hat)

fig, ax = plt.subplots ()
ax.set_title ('Log-likelihood scan', size=14)
ax.set_xlabel ('tau')
ax.set_ylabel ('log likelihood')

# choose a reasonable range around tau_hat, measured in terms of the confidence interval
xMin = tau_hat_minusS - (tau_hat - tau_hat_minusS)
xMax = tau_hat_plusS + (tau_hat_plusS - tau_hat)

tauaxis = np.linspace (xMin, xMax, 10000)
llr     = np.arange (0., tauaxis.size)
for i in range (tauaxis.size) :
    llr[i] = loglikelihood (tauaxis[i], exp_pdf, sample, tau_hat)

plt.plot (tauaxis, llr, 'r')
limits = ax.get_ylim ()
plt.plot ([tau_hat, tau_hat], limits, color = 'blue')
plt.plot ([tau_hat_minusS, tau_hat_minusS], limits, color = 'blue', linestyle = 'dashed')
plt.plot ([tau_hat_plusS, tau_hat_plusS], limits, color = 'blue', linestyle = 'dashed')
plt.plot (plt.xlim (), [-0.5, -0.5], color = 'gray')

plt.show ()
'''

#---------------------------
#4: TOY EXPERIMENT SULLO STIMATORE
#distribuzione di probabilità di tau
''''
Utilizzando il generatore di numeri pseudo-casuali secondo una pdf esponenziale sviluppato nelle lezioni passate,
si disegni la distribuzione di probabilita’ dello stimatore di τ in funzione del numero di eventi a disposizione per la stima.
'''

from math import floor
from stats import stats
    
#tau_true = 2.
N_evt    = 50
N_toys   = 1000

tau_hats = [] #lista in cui raccogliere i valori dei massimi (variano le iterazioni sulla funzione di maxLL)
# loop over toy experiments
for iToy in range (N_toys) :
    singleToy = generate_exp (tau_true, N_evt)
    tau_hat_toy = sezioneAureaMax_LL (loglikelihood, exp_pdf, singleToy, 0.5, 5., 0.0001)
    tau_hats.append (tau_hat_toy)

xMin = 1.
xMax = 3.
bin_edges = np.linspace (xMin, xMax, floor (N_toys/20))   # edges o the histogram bins

fig, ax = plt.subplots ()
ax.set_title ('Tau_hat expected distribution', size=14)
ax.set_xlabel('tau_hat')
ax.set_ylabel('events in bin')
ax.hist (tau_hats,
         bins = bin_edges,
         color = 'orange',
        )
'''
limits = ax.get_ylim ()
plt.plot ([tau_hat, tau_hat], limits, color = 'blue')
plt.plot ([tau_hat_minusS, tau_hat_minusS], limits, color = 'blue', linestyle = 'dashed')
plt.plot ([tau_hat_plusS, tau_hat_plusS], limits, color = 'blue', linestyle = 'dashed')
'''
toy_stats = stats (tau_hats)
print ('sigma ricavata dai toy:      ', toy_stats.sigma ())
#print ('sigma con il metodo grafico: ', 0.5 * (tau_hat_plusS - tau_hat_minusS))

plt.show ()


#5:ANDAMENTO ASINTOTICO

'''In regime asintotico, la distribuzione degli scarti (τ - τvero) / στ ha forma Normale.

    Si utilizzi il metodo dei toy experiment per riempire l’istogramma degli scarti, dato un numero di eventi per toy experiment.

    Si calcoli la media e la sigma della distribuzione degli scarti e se ne disegni il valore al variare del numero di eventi a disposizione per la stima, riempiendo un TGraph con il numero di eventi a disposizione sull’asse orizziontale ed il valore del parametro sull’asse verticale.
'''

N_evt_big = 200
sample_size = 5

N_events   = []
deviations = []
sigmas     = []
while sample_size <= N_evt_big :
    scarti = []
    # loop over toy experiments
    for iToy in range (N_toys) :
        singleToy = generate_exp (tau_true, sample_size)
        tau_hat_toy = sezioneAureaMax_LL (loglikelihood, exp_pdf, singleToy, 0.5, 5., 0.0001)
        scarti.append (tau_hat_toy - tau_true)
    toy_stats = stats (scarti)
    deviations.append (toy_stats.mean ())
    sigmas.append (toy_stats.sigma ())
    N_events.append (sample_size)
    sample_size = sample_size * 2

fig, ax = plt.subplots ()
ax.set_title ('average deviations', size=14)
ax.set_xlabel ('number of events')
ax.set_ylabel ('average deviation')
ax.errorbar (N_events, deviations, xerr = 0.0, yerr = sigmas) 
plt.show ()



#6: NARROWING
from likelihood import loglikelihood_ratio

sample_size = 10

fig, ax = plt.subplots ()
ax.set_title ('Log-likelihood ratio scan', size=14)
ax.set_xlabel ('tau')
ax.set_ylabel ('LLR')
tauaxis = np.linspace (0.5, 5., 10000)
colors = [
          'thistle',
          'plum',
          'violet',
          'orchid',
          'fuchsia',
          'mediumpurple',
          'purple',
         ]
index = 0
while sample_size <= N_evt :
    # create a subsample
    subsample = sample[:int (sample_size)]
    llr = np.arange (0., tauaxis.size)
    for i in range (tauaxis.size) :
        tau_hat = sezioneAureaMax_LL (loglikelihood, exp_pdf, subsample, 0.5, 5., 0.0001)    
        llr[i] = loglikelihood_ratio (tauaxis[i], exp_pdf, subsample, tau_hat)
    plt.plot (tauaxis, llr, color = colors[index], label = str (sample_size) + ' events')
    index = index + 1
    sample_size = sample_size + 10

plt.legend ()
plt.show ()