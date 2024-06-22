'''
Si generi un campione di numeri pseudo-casuali 
distribuiti secondo una distribuzione di densità esponenziale
con tempo caratteristico t0 di 5 secondi
e si visualizzi la distribuzione del campione ottenuto
in un istogramma utilizzando il metodo della funzione inversa.
Si scrivano tutte le funzioni deputate alla generazione di numeri casuali
in una libreria, implementata in file separati rispetto al programma principale

Si utilizzi il risultato del primo esercizio per simulare uno pseudo-esperimento di conteggio
con caratteristiche di Poisson:
  * si scelga un tempo caratteristico t0 di un processo di decadimento radioattivo;
  * si scelta un tempo di misura tm entro cui fare conteggi;
  * in un ciclo, si simulino N pseudo-esperimenti di conteggio, in cui,
    per ciascuno di essi, si simuli una sequenza di eventi casuali
    con intertempo caratteristico dei fenomeni di Poisson, 
    fino a che il tempo totale trascorso non sia maggiore del tempo di misura,
    contando il numero di eventi generati che cascano nell'intervallo;
  * si riempia un istogramma con i conteggi simulati per ogni esperimento  

Use the source code written in the previous exercise to add to the library developed for exercise 1 
a function that generates random numbers according to the Poisson distribution, with the mean expected events as an input parameter.

 * Rewrite the previous exercise using this function, also drawing the probability density histogram.

 * Calculate the sample statistics (mean, variance, skewness, kurtosis) from the input list using a library.

 * Use the generated sample to test the functionality of the library.



  * Si utilizzi il risultato dell'esercizio precedente
    per calcolare le statistiche di una distribuzione di Poisson 
    al variare della media, fra 1 e 250 (come conviene campionare l'intervallo?).
  * Si disegni l'andamento ottenuto della skewness e della curtosi in funzione della media

'''



import matplotlib.pyplot as plt
import numpy as np
from math import floor
import random
from math import sqrt


#!/usr/bin/python

from math import sqrt, pow


class stats :
    '''calculator for statistics of a list of numbers'''

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

    def skewness (self) :
        '''
        calculate the skewness of the sample present in the object
        '''
        mean = self.mean ()
        asymm = 0.
        for x in self.sample:
            asymm = asymm + pow (x - mean,  3)
        asymm = asymm / (self.N * pow (self.sigma (), 3))
        return asymm


    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def kurtosis (self) :
        '''
        calculate the kurtosis of the sample present in the object
        '''
        mean = self.mean ()
        kurt = 0.
        for x in self.sample:
            kurt = kurt + pow (x - mean,  4)
        kurt = kurt / (self.N * pow (self.variance (), 2)) - 3
        return kurt

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def append (self, x):
        '''
        add an element to the sample
        '''
        self.sample.append (x)
        self.summ = self.summ + x
        self.sumSq = self.sumSq + x * x
        self.N = self.N + 1
    #-------


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


def rand_poisson (mean) :
    '''
    generazione di un numero pseudo-casuale Poissoniano
    a partire da una pdf esponenziale
    '''
    total_time = rand_exp (1.)
    events = 0
    while (total_time < mean) :
        events = events + 1
        total_time = total_time + rand_exp (1.)
    return events


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_poisson (mean, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali Poissoniani
    a partire da una pdf esponenziale
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_poisson (mean))
    return randlist
#-------------
#--------------
#Si generi un campione di numeri pseudo-casuali  distribuiti secondo una distribuzione di densità esponenziale
#con tempo caratteristico t0 di 5 secondi
#e si visualizzi la distribuzione del campione ottenuto
#in un istogramma utilizzando il metodo della funzione inversa.
#----------------
def main () :
    '''
    Funzione che implementa il programma principale
    '''



    tau  = float (5)
    if tau <= 0 :
        print ('The tau parameter has to be positive')
        exit ()
    N    = int (10000)
    seed = 10
    
    randlist = generate_exp (tau, N, seed) #LISTA NUMERI RANDOM (GENERATI MEDIANTE F.INVERSA A DARE PDF EXPON)

    # plotting of the generated list of numbers in a histogram
    nBins = floor (len (randlist) / 100.)              # number of bins of the histogram
    bin_edges = np.linspace (0., 4. * tau, nBins + 1)  # edges o the histogram bins
    fig, ax = plt.subplots ()
    ax.set_title ('Exponential pdf with tau ' + str (tau), size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlist,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

    plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()


#--------------

# * si scelga un tempo caratteristico t0 di un processo di decadimento radioattivo;
# *si scelta un tempo di misura tm entro cui fare conteggi;

#* in un ciclo, si simulino N pseudo-esperimenti di conteggio, in cui,
# per ciascuno di essi, si simuli una sequenza di eventi casuali
#con intertempo caratteristico dei fenomeni di Poisson, 
#fino a che il tempo totale trascorso non sia maggiore del tempo di misura,
#contando il numero di eventi generati che cascano nell'intervallo;
# * si riempia un istogramma con i conteggi simulati per ogni esperimento

def main () :
    '''
    Funzione che implementa il programma principale
    '''

  
    tau_gen  = float (0.5)  #tempo di decadimento 
    tau_meas  = float (3) #tempo di misurazione
    if tau_meas < tau_gen :
        print ('The tau_meas parameter has to be larger than tau_gen')
       
    N    = int (10000)
    seed = 10
    
    randlist = [] #creo una rand list secondo rand_exp
    for i in range (N) :
        total_time = rand_exp (tau_gen) #TEMPO TOTALE dato il tau0
        events = 0
        while (total_time < tau_meas) : #confronto con il tempo di misura
            events = events + 1
            total_time = total_time + rand_exp (tau_gen) #in base a quanti conteggi rilevo (condizione è nel while) aggiungo il conteggio ad events
        randlist.append (events)    

    my_stats = stats (randlist)

    # plotting of the generated list of numbers in a histogram
    nBins = floor (len (randlist) / 100.)              # number of bins of the hitogram
    xMin = max (0., ceil (my_stats.mean () - 3 * my_stats.sigma ()))
    xMax = ceil (my_stats.mean () + 3 * my_stats.sigma ())
    bin_edges = np.linspace (xMin, xMax, int (xMax - xMin) + 1)  # edges o the histogram bins
    fig, ax = plt.subplots ()
    ax.set_title ('Counts distribution', size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlist,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

    plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()



#--------------------------------------------------------------
#*Use the source code written in the previous exercise to add to the library developed for exercise 1 
#a function that generates random numbers according to the POISSON DISTRIBUTION, with the mean expected events as an input parameter.

#* Rewrite the previous exercise using this function, also drawing the probability density histogram.

#* Calculate the sample statistics (mean, variance, skewness, kurtosis) from the input list using a library.

#* Use the generated sample to test the functionality of the library.

#!/usr/bin/python



import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil

#from myrand import generate_poisson
#from stats import stats


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    mean = float (10)
    if mean <= 0 :
        print ('The mean parameter has to be positive')
        exit ()
    N    = int (1000)
    seed = 10

    randlist = generate_poisson (mean, N) #ECCOLO QUI

    my_stats = stats (randlist)

    print ('mean      : ', mean, my_stats.mean ())
    print ('variance  : ', mean, my_stats.variance ())

    # plotting of the generated list of numbers in a histogram
    nBins = floor (len (randlist) / 100.)              # number of bins of the hitogram
    xMin = max (0., ceil (my_stats.mean () - 3 * my_stats.sigma ()))
    xMax = ceil (my_stats.mean () + 3 * my_stats.sigma ())
    bin_edges = np.linspace (xMin, xMax, int (xMax - xMin) + 1)  # edges o the histogram bins
    fig, ax = plt.subplots ()
    ax.set_title ('Counts distribution', size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlist,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

    plt.show() ('es_7.3.png')


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()



#---------------------------------------------
#  * Si utilizzi il risultato dell'esercizio precedente
#   per calcolare le statistiche di una distribuzione di Poisson 
#   al variare della media, fra 1 e 250 (come conviene campionare l'intervallo?).
# * Si disegni l'andamento ottenuto della skewness e della curtosi in funzione della media
#!/usr/bin/python



import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil



def main () :
    '''
    Funzione che implementa il programma principale
    '''


    mean_min = float (1)
    mean_max = float (250)

    N    = int (1000)
    seed = 10

    means = []
    skewn = []
    kurts = []
    while (mean_min < mean_max) :
        randlist = generate_poisson (mean_min, N) 
        my_stats = stats (randlist)
        means.append (my_stats.mean ())
        skewn.append (my_stats.skewness ())
        kurts.append (my_stats.kurtosis ())
        mean_min *= 2  #ecco come conviene campionare

    fig, (ax1, ax2) = plt.subplots (2, sharex=True)
    fig.suptitle ('Poisson distribution asymptotic behaviour')
    ax1.plot (means, skewn)
    ax1.set_ylabel ('skewness')
    ax2.plot (means, kurts)
    ax2.set_ylabel ('kurtosis')
    ax2.set_xlabel ('mean')

    plt.show()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()