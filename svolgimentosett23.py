'''
1. Si implementi una libreria che generi punti pseudo-casuali distribuiti secondo una distribuzione
di Cauchy in un intervallo simmetrico attorno ad a utilizzando il metodo try-and-catch TAC, dove la
funzione f_cauchy che produce ciascun numero casuale prenda come parametri in ingresso i due
parametri, a e b e la semi-larghezza dell’intervallo di generazione.

2. Si scriva un programma main.cpp che verifichi il funzionamento della libreria generando N numeri
casuali fra (a − 3b) ed (a + 3b), prendendo i valori di N , a e b come parametri in ingresso a
linea di comando al momento della chiamata del programma.

3. Si accumulino questi numeri in un istogramma di tipo TH1F con minimo, massimo e numero di bin
scelti algoritmicamente sulla base di N , a e b e lo si disegni in un’immagine di tipo png.

4. Si calcolino media e sigma di numeri pseudo-casuali generati secondo una distribuzione di Cauchy
sull’intervallo (a − ib, M + ib) al variarare di iΓ fra 1 e 100 e si rappresenti l’andamento delle
due quantità in funzione di ib su due TGraph, disegnati in due immagini di tipo png.

5. Si implementi una funzione rand_TCL_cauchy che generi numeri pseudo-casuali con la tecnica del
teorema centrale del limite a partire da numeri generati con la funzione f_cauchy. Si trovi un modo
di verificare se i numeri generati con rand_TCL_cauchy siano effettivamente distribuiti secondo una
Gaussiana: lo sono? Perché?
'''
import numpy as np
from math import pi, ceil, floor
import random
from math import sqrt
import matplotlib.pylab as plt
from stats import stats 

def f_cauchy(x,a,b):
    c = 1/np.pi
    q = (x-a)*(x-a)
    return c*(b/(q+b*b))


def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TAC_cauchy (a,b,xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    yMax = f_cauchy(a,a,b) #il massimo si trova derivata=0 da cui x=a
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    
    a=1
    b=1
    while (y > f_cauchy (x,a,b)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TAC_cauchy (a,b,xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range(N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TAC_cauchy (a,b,xMin, xMax))
    return randlist

#-------
#-------
def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )
#-----
#-----
def rand_TCL_cauchy (a,b,xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_TAC_cauchy (a,b,xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL (a,b,xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append(rand_TCL_cauchy(a,b,xMin, xMax, N_sum))
    return randlist


#int main():

#intervallo simmetrico: a-k, a+k
#input per sample cauchy TAC:
a= 1
b= 1
xMin = a-3*b
xMax = a+3*b
yMax= (1/np.pi)*(1/b)
N = int(100)
#sample1=generate_TAC_cauchy(a,b,xMin, xMax,yMax, N)
#print(sample1)


#k=1,..,1000


data_means = []
data_sigmas = []
X = []
for k in range(1,100):

    xMin = a-k*b
    xMax = a+k*b
    sample1 = generate_TAC_cauchy(a,b,xMin, xMax,yMax,N)
    my_toy = stats(sample1)
    data_means.append(my_toy.mean())
    data_sigmas.append(my_toy.sigma())
    X.append(k)

#print(data_means)
#print(data_sigmas)

fig, axes = plt.subplots (1,2)

axes[0].plot (X, data_means)
axes[0].set_title ("andamento delle medie al variare dell'intervallo di acquisizione||", size=6)
axes[1].plot(X, data_sigmas)
axes[1].set_title ("andamento delle sigma al variare dell'intervallo di acquisizione", size=6)
plt.show()


xx = np.array([10,100,1000,2000])


sample2_1=(generate_TCL (a,b,xMin, xMax, xx[0], N_sum = 10, seed = 0.)) 
sample2_2=(generate_TCL (a,b,xMin, xMax, xx[1], N_sum = 10, seed = 0.)) 
sample2_3=(generate_TCL (a,b,xMin, xMax, xx[2], N_sum = 10, seed = 0.)) 
sample2_4=(generate_TCL (a,b,xMin, xMax, xx[3], N_sum = 10, seed = 0.)) 

N_bins_1=int(max(len(sample2_1),len(sample2_2))/2.)
N_bins_2=int(max(len(sample2_3),len(sample2_4))/10.) #usare in generale sturges

N_bins_optimized=sturges(len(sample2_4)) 

fig, axes = plt.subplots(2,2) #qui avrò una griglia 2x2 con quattro grafici

axes[0,0].hist(sample2_1,
              bins=N_bins_1,
              label='eventi unif',
              color='pink'
              )
#axes[0,0].set_ylim([0,40]) #cosi rendo confrontabili i grafici: il range prima andava da 0 a 20, 0-40 è il range di axes[0,1] (data2)

axes[0,1].hist(sample2_2,
              bins=N_bins_1,
              label='eventi gauss',
              color='blue'
              )


axes[1,0].hist(sample2_3,
              bins=N_bins_2,
              color='cyan'
              )
#axes[1,0].set_ylim([0,800])


axes[1,1].hist(sample2_4,
              bins=N_bins_2,
              label='eventi gauss',
              color='hotpink'
              )

plt.show()

''''
N_bins = sturges (len (sample1)/1.)
bin_edges = np.linspace (xMin/2., xMax/2.+1., N_bins)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample1,
         bins = bin_edges,
         color = 'orange',
        )
plt.show()
'''