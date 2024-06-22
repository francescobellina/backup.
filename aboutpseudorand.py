#generatori di numeri pseudocasuali e algoritmi per numeri pseudocasuali secondo distribuioni varie



#0: GENENERAZIONE NUMERI PC SECONDO UNA DISTRIBUZIONE UNIFORME
'''
Si implementi un generatore di numeri pseudo-casuali secondo una distribuzione uniforme
fra due estremi arbitrari.

'''

import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor

from myrand import generate_range


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    if len(sys.argv) < 5 :
        print ('usage: ', sys.argv[0], 'xMin xMax numero seed')
        exit ()

    xMin = float (sys.argv[1])  # minimum of the histogram drawing range
    xMax = float (sys.argv[2])  # maximum of the histogram drawing range
    seed = float (sys.argv[4])
    N    = int (sys.argv[3])


    randlist = generate_range (xMin, xMax, N, seed)

    # plotting of the generated list of numbers in a histogram

    nBins = floor (len (randlist) / 20.) + 1     # number of bins of the hitogram
    bin_edges = np.linspace (xMin, xMax, nBins)  # edges o the histogram bins

    # disegno della funzione
    fig, ax = plt.subplots ()
    ax.set_title ('Histogram of random numbers', size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlist,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()


#1: IMPLEMENTAZIONE TAC ALGORITHM

'''
generazione di numeri pseudo-casuali distribuiti secondo una distribuzione arbitraria
con il metodo try-and-catch fra xMin ed xMax
a partire da un determinato seed e disegno della distribuzione
'''

import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor

from myrand import generate_TAC

def func (x, scale = 1) : 
    '''
    funzione definita positiva sotto la quale generare numeri casuali
    '''
    return scale * (scale*np.sin(x)+1)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def main () :
    '''
    Funzione che implementa il programma principale
    '''



    xMin = float (-3)  # minimum of the histogram drawing range
    xMax = float (3.)  # maximum of the histogram drawing range
    yMax = float (1.)  # maximum of the histogram drawing range    
    seed = float (1)
    N    = int (800)

 
    randlist = generate_TAC (func, xMin, xMax, yMax, N, seed)

    # plotting of the generated list of numbers in a histogram

    nBins = floor (len (randlist) / 400.)             # number of bins of the hitogram
    bin_edges = np.linspace (xMin, xMax, nBins + 1)  # edges o the histogram bins

    # disegno della funzione
    fig, ax = plt.subplots ()
    ax.set_title ('Histogram of random numbers', size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlist,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

    plt.show ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()




#2: IMPLEMENTAZIONE TCL



def main () :
    '''
    Funzione che implementa il programma principale
    '''



    xMin = float (0)  # minimum of the histogram drawing range
    xMax = float (100)  # maximum of the histogram drawing range
    yMax = float (10)  # maximum of the histogram drawing range    
    seed = float (1)
    N    = int (800)
    N_sum = 10
 

    randlist = generate_TCL (xMin, xMax, N, N_sum, seed)

    # plotting of the generated list of numbers in a histogram

    nBins = floor (len (randlist) / 400.)             # number of bins of the hitogram
    bin_edges = np.linspace (xMin, xMax, nBins + 1)  # edges o the histogram bins

    # disegno della funzione
    fig, ax = plt.subplots ()
    ax.set_title ('Histogram of random numbers', size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlist,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

    plt.show ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()


#3:FUNZIONE INVERSA (exponential prob)
'''
Si implementi un generatore di numeri pseduo-casuali che utilizzi il metodo della funzione inversa
AL FINE DI  generare eventi distribuiti secondo distribuzione di probabilita' esponenziale.
(il cambio di variabili evidenzia ciò(proprietà), se sulla funzione inversa ho delle y distribuite casualmente e uniformemente allora avrò delle x 
distibuite in maniera
casuale secondo una pdf esponenziale)
'''


import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor




def inv_exp (y, lamb = 1) : #PARTO dalla funzione inversa della pdf che mi serve iniziando generare sulle y i numeri pc generati unif
    '''
    Inverse of the primitive of the exponential PDF.
    pdf(x) = lambda * exp(-lambda x) x >= 0, 0 otherwise.
    F(x) = int_{0}^{x} pdf(x)dx = 1 - exp(-lambda * x) for x >= 0, 0 otherwise.
    F^{-1}(y) = - (ln(1-y)) / lambda
    '''
    return -1 * np.log (1-y) / lamb


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    tau  = float (0.5) #il tau inserito deve essere >= 0
    lamb = 1./tau ;    
    N    = int (1000)
    seed = 1

    lamb = 1./tau ; 
    random.seed (seed)

    randlist = []
    for i in range (N):
        randlist.append (inv_exp (random.random (), lamb)) #genero un array su la funzione inversa con argomento numeri pc tra 0 e 1 e lamb

    # plotting of the generated list of numbers in a histogram

    nBins = floor (len (randlist) / 100.)        # number of bins of the hitogram
    bin_edges = np.linspace (0., 3., nBins + 1)  # edges o the histogram bins

    # disegno della funzione
    fig, ax = plt.subplots ()
    ax.set_title ('Histogram of random numbers', size=14)
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




#TCL E USO DI STATS:
'''

si implementi un oggetto chiamato ```stats```, 
che calcola le statistiche associate ad un campione di numeri
salvato in una lista passata al TCL_ms
  * quali diverse opzioni di progettazione sono possibili per questo oggetto?
  * che variabili è necessario aggiungere alla classe per garantirne la funzionalità?
  * che valore devono avere queste variabili in fase di inizilizzazione?
Si collaudi l'oggetto ```stats``` con ciascuono degli agoritmi di generazione implementati.
In particolare, poi:
  * si verifichi che il valore della varianza della distribuzione uniforme corrisponde alle attese
    (quale è l'incertezza associata al numero ottenuto?)
  * si verifichi che il valore della varianza ottenuta con la tecnica del teorema centrale del limite
    corrisponda a quello atteso
    
'''


import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor

from myrand import generate_TCL_ms
from stats import stats


def main () :
    '''
    Funzione che implementa il programma principale
    '''

    if len(sys.argv) < 6:
        print ('usage: ', sys.argv[0], 'mean sigma numero seed numero_somme')
        exit ()

    mean  = float (sys.argv[1])  # minimum of the histogram drawing range
    sigma = float (sys.argv[2])  # maximum of the histogram drawing range
    seed  = float (sys.argv[4])
    N     = int (sys.argv[3])
    N_sum = int (sys.argv[5])

    randlistTCL = generate_TCL_ms (mean, sigma, N, N_sum, seed) #TCL indicando anche sigma e mean

#OGGETTO my_stats sulla lista creata attraverso l'algoritmo TCL_ms
    my_stats = stats (randlistTCL)
    my_mean = my_stats.mean ()
    my_sigma = my_stats.sigma ()
    print ('media:    ', my_mean) ;
    print ('varianza: ', my_stats.variance ()) ;
    print ('sigma:    ', my_sigma) ;

    # plotting of the generated list of numbers in a histogram

    nBins = floor (len (randlistTCL) / 400.)             # number of bins of the hitogram
    bin_edges = np.linspace (mean - 3 * sigma, mean + 3 * sigma, nBins + 1)  # edges o the histogram bins

    # disegno della funzione
    fig, ax = plt.subplots ()
    ax.set_title ('Histogram of random numbers', size=14)
    ax.set_xlabel ('random value')
    ax.set_ylabel ('events in bin')
    ax.hist (randlistTCL,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )
    #ORA PLOTTO ANCHE I VALORI OTTENUTI DALLA CLASSE STATS
    yrange = plt.ylim ()
    ax.plot ([my_mean, my_mean], yrange, color = 'red') 
    ax.plot ([my_mean - my_sigma, my_mean - my_sigma], yrange, linestyle='dashed', color = 'red') 
    ax.plot ([my_mean + my_sigma, my_mean + my_sigma], yrange, linestyle='dashed', color = 'red') 

    plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()