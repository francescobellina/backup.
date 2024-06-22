#TOY-EXPERIMENTS

#!/usr/bin/python
'''
generazione di N_toy toy experiment contenenti ciascuno N_evt campioni,
ciascuno dei quali segue una distribuzione uniforme fra 0 ed 1

'''

from myrand import generate_uniform


N_toy = 3       #3 esperimenti ognuno da 10 campionamenti (naturalmente far crescere sia toy che evt)
N_evt = 10
 
# loop over toys
for i in range (N_toy):
    randlist = generate_uniform (N_evt) #(questo è di per se un array con numeri generati casualmente)
    print("\nIl n", i , "toy exp con", N_evt , "misurazioni vale:" , randlist,"\n")



#--------------------------------------------
#!/usr/bin/python
'''
generazione di N_toy toy experiment contenenti ciascuno N_evt campioni,
ciascuno dei quali segue una distribuzione uniforme fra 0 ed 1
e disegno della distribuzione delle medie dei campioni al variare dei toy

'''

import numpy as np
from math import floor
import matplotlib.pyplot as plt

from myrand import generate_uniform
from stats import stats #libreria stats scritta come classe.



N_toy = 100
N_evt = 1000
 
means = []
sigma_means = []

# loop over toys
for i in range (N_toy):
    randlist = generate_uniform (N_evt)
    toy_stats = stats (randlist)    #chiamata alla classe stats
    means.append (toy_stats.mean ())  #chiamata alla funzione mean di stats tramite l'attributo toy_stats, aggiungo all'arrey progressivo le medie
    sigma_means.append (toy_stats.sigma_mean ())

    
# compare the distribution of the sigma on the mean (deviazione std della media)
# calculated for each toy to the sigma of the mean distribution
means_stats = stats (means)   #chiamata alla classe, itero il processo-->voglio mediare tutte le medie ottenute per i singoli toy
sigma_means_stats = stats (sigma_means)


print ('sigma of the means disitribution:             ', means_stats.sigma ()) #dev std delle medie
print ('mean of the sigma of the means disitribution: ', sigma_means_stats.mean ()) #media delle dev std


# plot the distribution of the sigma on the mean
# calculated for each toy
xMin = sigma_means_stats.mean () - 5. * sigma_means_stats.sigma ()
xMax = sigma_means_stats.mean () + 5. * sigma_means_stats.sigma () 
nBins = floor (len (sigma_means) / 10.) + 1     # number of bins of the histogram
bin_edges = np.linspace (xMin, xMax, nBins + 1)  # edges o the histogram bins
fig, ax = plt.subplots ()
ax.set_title ('Histogram of the sigma on the mean over ' + str (N_toy) + ' toys', size=14)
ax.set_xlabel ('mean value')
ax.set_ylabel ('toys in bin')
ax.hist (sigma_means,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

# get the sigma of the means distribution VISUALIZZAZIONE DELL'ERRORE SULL MEDIA
ax.plot ([means_stats.sigma (), means_stats.sigma ()], plt.ylim ())
plt.show ()



#--------------
#ESERCIZI ED ESEMPI:

#1:
#Si utilizzino due scatter plot per confrontare l'evoluzione
#della deviazione standard della media calcolata per ogni singolo toy
#con la deviazione standard del campione delle medie
#al variare del numero di eventi generati in un singolo toy experiment.
#AL VARIARE ANCHE DEGLI N_EVENTI, SCALE LOGARITMICHE

import numpy as np
from math import floor
import matplotlib.pyplot as plt

from myrand import generate_uniform
from stats import stats

def main () :
    '''
    Funzione che implementa il programma principale
    '''

    

    N_toys = int (1000)
    N_evt_min = int (10)
    N_evt_max = int (1010)

    # deviazione standard del campione delle medie
    means_sigma = []
    # single sigma of the mean
    sigma_of_the_mean = []
    sigmas_of_the_mean = []
    N_events = [] 
    X = []
    while (N_evt_min < N_evt_max) :
        means = []
        # loop over toys
        for i in range (N_toys):
            randlist = generate_uniform (N_evt_min)
            toy_stats = stats (randlist)
            means.append (toy_stats.mean ())
            if i == 0 : sigma_of_the_mean.append (toy_stats.sigma_mean ())
            sigmas_of_the_mean.append (toy_stats.sigma_mean ())
            X += [N_evt_min]
        N_events.append (N_evt_min)
        toy_stats = stats (means)
        means_sigma.append (toy_stats.sigma ())
        N_evt_min = N_evt_min * 2    

    fig, ax = plt.subplots ()
    ax.set_title ('sigma of the means', size=14)
    ax.set_xlabel ('number of events')
    ax.set_ylabel ('sigma of the mean')
    ax.scatter(X,sigmas_of_the_mean,label='stddev of the mean')
    ax.plot (N_events, means_sigma, color = 'red', label = 'all toys')
    ax.plot (N_events, sigma_of_the_mean, color = 'blue', label = 'single toy')
    ax.set_xscale ('log')
    ax.legend ()

    plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()


#2: HOM
''''
Si implementi il metodo di integrazione hit-or-miss
con la funzione di esempio f(x) = sin (x)
  * Si scriva l'algoritmo che calcola l'integrale come una funzione esterna al programma ```main```,
    facendo in modo che prenda come parametri di ingresso,
    oltre agli estremi lungo l'asse x e l'asse y,
    anche il numero di punti pseudo-casuali da generare.
  * Si faccia in modo che l'algoritmo ritorni due elementi:
    il primo elemento sia il valore dell'integrale,
    il secondo sia la sua incertezza.

Si inserisca il calcolo dell'integrale dell'esercizio precedente in un ciclo che,
al variare del numero N di punti generati, mostri il valore dell'integrale
e della sua incertezza.
  * Si utilizzi uno scatter plot per disegnare gli andamenti del valore dell'integrale
    e della sua incertezza, al variare di N con ragione logaritmica.
'''



def integral_HOM (func, xMin, xMax, yMax, N_evt) :



def func (x) : 
    return 1. + np.sin (x) ; 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def main () :
    '''
    Funzione che implementa il programma principale
    '''


    xMin = float (0)
    xMax = float (6)
    yMax = float (1)
    N_evt_min = int (10)
    N_evt_max = int (10000)
 
    N_events = []
    integrals = []
    integral_errors = []
    while (N_evt_min < N_evt_max) :
        integral, integral_unc = integral_HOM (func, xMin, xMax, yMax, N_evt_min) #<---
        integrals.append (integral)
        integral_errors.append (integral_unc)
        N_events.append (N_evt_min) # <--
        N_evt_min = N_evt_min + 50

    fig, ax = plt.subplots ()
    ax.set_title ('integral precision', size=14)
    ax.set_xlabel ('number of events')
    ax.set_ylabel ('integral and its error')
    #ax.scatter(N_events, integrals, s = 1)
    #ax.scatter(N_events, integral_errors, s = 1)
    ax.errorbar (N_events, integrals, xerr = 0.0, yerr = integral_errors)
    plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()


#3:CRUDE MC
'''
Si implementi il metodo di integrazione crude-MC
con la funzione di esempio f(x) = sin (x).
  * Si scriva l'algoritmo che calcola l'integrale come una funzione esterna al programma ```main```,
    facendo in modo che prenda come parametri di ingresso,
    oltre agli estremi lungo l'asse *x*,
    anche il numero di punti pseudo-casuali da generare.
  * Si faccia in modo che l'algoritmo ritorni un contenitore contenente due elementi:
    il primo elemento sia il valore dell'integrale,
    il secondo sia la sua incertezza.


Si inserisca il calcolo dell'integrale dell'esercizio precedente in un ciclo che,
al variare del numero *N* di punti generati, mostri il valore dell'integrale
e della sua incertezza.
  * Si disegnino gli andamenti del valore dell'integrale
    e della sua incertezza, al variare di *N* con ragione logaritmica.
**Si sovrapponga questo andamento a quello ottenuto dallo svolgimento dell'Esercizio con HOM.

'''

def integral_HOM (func, xMin, xMax, yMax, N_evt) :
   
  
                                         
def func (x) : 
    return 1. + np.sin (x) ; 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def main () :
    '''
    Funzione che implementa il programma principale
    '''


    xMin = float (0)
    xMax = float (6)
    yMax = float (1)
    N_evt_min = int (10)
    N_evt_max = int (1000)
    
    N_events = []
    integrals_HOM = []
    integral_errors_HOM = []
    integrals_MC = []
    integral_errors_MC = []

    while (N_evt_min < N_evt_max) :

        integral, integral_unc = integral_CrudeMC(func, xMin, xMax, N_evt_min) 
        integrals_MC.append (integral)
        integral_errors_MC.append (integral_unc)
        
        integral, integral_unc = integral_HOM(func, xMin, xMax, yMax, N_evt_min)
        integrals_HOM.append (integral)
        integral_errors_HOM.append (integral_unc)
        
        N_events.append (N_evt_min) 
        N_evt_min = N_evt_min*2

    fig, ax = plt.subplots ()
    ax.set_title ('integral precision CRUDE MC vs HOM methods', size=14)
    ax.set_xlabel ('number of events')
    ax.set_ylabel ('integral and its error')
    
    print(len(N_events)) 
    print(len(integrals_MC))
    ax.errorbar (N_events, integrals_MC, xerr = 0.0, yerr = integral_errors_MC, color = "blue")
    ax.errorbar (N_events, integrals_HOM, xerr = 0.0, yerr = integral_errors_HOM, color = "red")
    ax.legend()
    plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()


#4: HOM  su una GAUSSIANA

'''Si utilizzi il metodo hit-or-miss per stimare l’integrale sotteso
ad una distribuzione di probabilita' Gaussiana con mu=0 e sigma=1
in un generico intervallo [a,b]
  * Si calcoli l'integrale contenuto entro gli intervalli [-ksigm,ksigma]
    al variare di k da 1 a 5.
'''

from scipy.stats import norm


def normal(x):
    return norm.pdf (x, 0., 1.)



xMin=-1
xMax=1
yMax= (1/(sqrt(2*np.pi)))
N_evt=1000

I_value, I_prec = integral_HOM(normal,xMin, xMax, yMax, N_evt)
print(I_prec)
for i in range(6):
    xMin= -i
    xMax= i
    I_value, I_sigma = integral_HOM(normal,xMin, xMax, yMax, N_evt)
    if i==0: continue
    else:
        print("l'area della gaussiana sottesa in un intervallo a +-",i,"sigma vale: ", I_value)
