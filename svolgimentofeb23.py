'''
*Si prepari una funzione rand_TCL_unif, che generi numeri pseudo-casuali distribuiti secondo una
distribuzione di densità di probabilità Gaussiana nell’intervallo (1, 3) utilizzando il teorema centrale
del limite TCL, a partire da numeri pseudo-casuali distribuiti secondo una distribuzione di densità di
probabilità uniforme, 
*si scriva un main.cpp dove vengano generati 10000 numeri con questa fun-
zione, li si disegni in un file di tipo png tramite un istogramma 
'''
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from stats import stats

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)



def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum 
    return y 


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

#-------
#------
#PARAbola
def rand_TCL_para (xMin, xMax, yMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_TAC (parab,xMin, xMax, yMax) #L'ALGORITMO TAC MI PERMETTE DI GENERARE NUMERI PC SECONDA UNA CERTA f
    y /= N_sum 
    return y 

def generate_TAC_para (xMin, xMax, yMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL_para (xMin, xMax, yMax))
    return randlist



def parab(x):
    return -(x-2)*(x-2) + 1
#------------------



sample1 = generate_TCL (1.,3., 10000, N_sum = 10, seed = 0.)
sample2 = generate_TAC (parab, 1., 3., 1., 10000, seed = 0.) 
sample3= generate_TAC_para (1., 3., 1., 10000, seed = 0.)

tac_uniform = stats(sample2)
tac_parab = stats(sample3)
print("La media sul campione generato a partire da numeri pc uniformi vale: ", tac_uniform.mean())
print("La sigma sul campione generato a partire da numeri pc uniformi vale: ", tac_uniform.sigma())
print("La kurtosi sul campione generato a partire da numeri pc uniformi vale: ", tac_uniform.kurtosis())
print("La skewness sul campione generato a partire da numeri pc uniformi vale: ", tac_uniform.skewness())
print("--------------------------------------------------------------------------------------")
print("La media sul campione generato a partire da numeri pc parabolici vale: ", tac_parab.mean())
print("La sigma sul campione generato a partire da numeri pc parabolici vale: ", tac_parab.sigma())
print("La kurtosi sul campione generato a partire da numeri pc parabolici vale: ", tac_parab.kurtosis())
print("La skewness sul campione generato a partire da numeri pc parabolici vale: ", tac_parab.skewness())

def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

N_bins = sturges (len (sample1))
bin_edges = np.linspace (0., 3., N_bins)

fig, axes = plt.subplots (1,3)
axes[0].hist (sample1,
         bins = bin_edges,
         color = 'orange',
        )
axes[1].hist (sample2,
         bins = bin_edges,
         color = 'green',
        )

axes[2].hist (sample3,
         bins = bin_edges,
         color = 'red',
        )
axes[0].set_xlim (0.9 , 3.1)
axes[1].set_xlim (0.9 , 3.1)
axes[2].set_xlim (0.9 , 3.1)
plt.show()

#ora procediamo ai 10000 toy-experiment da cui estrarre i valori di skewness e kurtosi
N_toys = int (1000)
N_evt_min = int (10)
N_evt_max = int (1000)

# deviazione standard del campione delle medie
means_sigma = []
    # single sigma of the mean
skew_1= []
kurt_1 = []
skew_2 = []
kurt_2 = []
N_events = [] 
X = []
while (N_evt_min < N_evt_max) :
    means = []
     # loop over toys
    for i in range (N_toys):
        randlist1 = generate_TAC (parab, 1., 3., 1., N_evt_min, seed = 0.) 
        toy_stats = stats (randlist1)
        skew_1.append (toy_stats.skewness ())
        kurt_1.append (toy_stats.kurtosis ())

        randlist2 = generate_TAC_para (1., 3., 1., N_evt_min, seed = 0.)
        toy_stats2 = stats (randlist2)
        skew_2.append (toy_stats2.skewness ())
        kurt_2.append (toy_stats2.kurtosis ())

        X += [N_evt_min]

    N_events.append (N_evt_min)
    N_evt_min = N_evt_min*2  

fig, ax = plt.subplots ()
    
ax.plot(X,skew_1, s=1)
ax.plot(X,skew_2, s=1, c='red')
#ax.set_xscale ('log')
ax.plot(X,kurt_1, s=1)
ax.plot(X,kurt_2, s=1, c='red')
plt.show()


'''
Si aggiunga alla libreria del punto precedente una funzione rand_TAC che generi numeri pseudo-
casuali nell’intervallo (1, 3) distribuiti secondo la seguente distribuzione di densità di probabilità
parabolica f (x):
f (x) = −(x − 2)^2 + 1 ;
nel programma principale main.cpp la generazione di 10000 numeri con questa funzione, li si dis-
egni in un file di tipo png tramite un istogramma.
'''

'''Si aggiunga alla libreria del primo punto una funzione rand_TCL_para che generi numeri pseudo-
casuali nell’intervallo (1, 3) distribuiti secondo una distribuzione di densità di probabilità Gaussiana
utilizzando il teorema centrale del limite, a partire da numeri pseudo-casuali DISTRIBUITI SECONDO
f (x) e li si disegni in un file di tipo png come nel punto precedente.

--> TAC(parab(x)) in TCL
'''

'''
Si crei una nuova libreria che calcoli asimmetria e curtosi di un campione di eventi, e si calcolino queste due quantità per un campione di 10000 eventi generato con
rand_TCL_unif ed per uno generato con rand_TCL_para.

Utilizzando quattro TGraph, si traccino in due file di tipo png l’andamento di asimmetria e curtosi
per campioni generati con rand_TCL_unif e rand_TCL_para rispettivamente, al variare del numero
di eventi pseudo-casuali generati all’interno di queste due ultime funzioni. Dopo quanti numeri le
due funzioni possono essere considerate equivalenti in termini di prestazioni? Perché?
'''