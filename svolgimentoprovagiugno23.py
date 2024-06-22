
import random
from math import sqrt
import numpy as np
from math import pi, e, exp, cos, floor,ceil
import matplotlib.pyplot as plt
from stats import stats
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


def integral_HOM (func, xMin, xMax, yMax, N_evt) :
    '''
    L’algoritmo hit-or-miss si comporta in modo simile alla generazione di numeri pseudo-casuali con la tecnica try-and-catch
    Si generano N coppie numeri pseudo-casuali nel piano che contiene il disegno della funzione
    e si conta il numero di eventi nhit che cascano nell’area sottesa dalla funzione 
    '''

    x_coord = generate_range (xMin, xMax, N_evt)
    y_coord = generate_range (0., yMax, N_evt)

    points_under = 0
    for x, y in zip (x_coord, y_coord):
        if (func (x) > y) : points_under = points_under + 1 

    A_rett = (xMax - xMin) * yMax
    frac = float (points_under) / float (N_evt)
    integral = A_rett * frac
    integral_unc = A_rett**2 * frac * (1 - frac) / N_evt #incertezza secondo la varianza derivante dalla natura binomiale di nhit (points_under)
    return integral, integral_unc #restituisce il valore dell'integrale con la sua precisione. Nota I è anch'esso un numero pseudo casuale



N_pc = int(10)

#X_rand = generate_range (0., e , N_pc, seed = 0.)
#Y_rand = generate_range (-np.pi/2, np.pi/2 , N_pc, seed = 0.)
#Z_rand = generate_range (0., 1, N_pc, seed = 0.)

def volume_rand(N):
    x = generate_range (0., e , N, seed = 0.)
    y = generate_range (-np.pi/2, np.pi/2 , N, seed = 0.)
    z = generate_range (0., 1, N, seed = 0.)
    
    randlist = []
    for i in range(N):
        randlist.append(x[i]*y[i]*z[i])

    return randlist    

#print(volume_rand(10))

def func2(y):
    return np.cos(y)

def func1(x):
    return np.exp(-x)


def calc_integral(up_x, bott_x, up_y, bott_y, zmax, N): #2-2/exp(e) circa 1.8680
    x, xe = integral_HOM (func1, up_x, bott_x, zmax, N)
    y, ye= integral_HOM (func2, up_y, bott_y, zmax, N)
    integral= x*y
    integral_err = xe+ye   
    return integral, integral_err



N_iterat = 1000
N_toys = 100

I_list = []
I_err_list = []
for i in range(N_toys):
    s,se = calc_integral(0.,e,-np.pi/2,np.pi/2,1,N_iterat)
    I_list.append(s)
    I_err_list.append(se)


#print("la sequenza dei valori dell'integrale è: ", I_list)
#print("gli errori sono: ", I_err_list)


my_mean=stats(I_list)
my_sigma=stats(I_err_list)
media = my_mean.mean()
sigma = my_sigma.sigma_mean()
print("la media degli integrali sui 100 toy-exp è:", media, "con sigma pari a: ", sigma)

xMin = media -   5500*sigma 
xMax = media  +   5500*sigma


#nBins = floor (len (I_list)/10.) + 1     # number of bins of the histogram
#bin_edges = np.linspace (xMin, xMax, nBins + 1)  # edges o the histogram bins
def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

N_bins = sturges (len (I_list)/1)
bin_edges = np.linspace (xMin, xMax, N_bins)
fig, ax = plt.subplots ()
ax.set_title ('istogramma dei valori dellintegrale per'+ str (N_toys) + ' toy experiment', size=14)
ax.set_xlabel ('integral value')
ax.set_ylabel ('toys in bin')
ax.hist (I_list,      # list of numbers
             bins = bin_edges,
             color = 'orange',
             # normed = True,
            )

ax.plot ([media, media], plt.ylim ())
plt.axvline(x = media + sigma , color = 'red', ls='--')
plt.axvline(x = media - sigma , color = 'red', ls='--')
#plt.show()

#Si confronti l’incertezza calcolata dall’algoritmo hit-or-miss con quella deducibile dall’istogramma e
#si commenti il confronto fra i due numeri:

#dopo aver trovato un binnaggio adeguato notiamo che l'incertezza derivante dall'algoritmo sia decisamente più
#piccola rispetto al size del bin e all'andamento medio del grafico, per osservarla nel grafico bisogna
#amplificarla per evitare che collassi, graficamente, nel valor medio.


'''
Si studi l’evoluzione del valore e della precisione nel calcolo dell’integrale in funzione del numero
di eventi pseudo-casuali generati rappresentandola in un TGraphErrors di ROOT, utilizzando appro-
priatamente scale logaritmiche. Si produca un’immagine di tipo png dell’andamento ottenuto.
'''

N_toys = int (100)
N_evt_min = int (10)
N_evt_max = int (10000)


x_ascisse = [] 
mean_value = []
sigma_value = []
means_sigma = []

while (N_evt_min < N_evt_max) :
    integrals = []
    
    for i in range (N_toys):
        s,se = calc_integral(0.,e,-np.pi/2,np.pi/2,1,N_evt_min)
        I_list.append(s)
        I_err_list.append(se)
        my_mean = stats(I_list)
        my_sigma = stats(I_err_list)
        mean_value.append(my_mean.mean())
        sigma_value.append(my_sigma.sigma())
        x_ascisse.append(i)
    
    toy_stats = stats (mean_value)
    means_sigma.append (toy_stats.mean ())
    toy_stats_sigma = stats(sigma_value)
    means_sigma.append (toy_stats.sigma ())
    N_evt_min = N_evt_min * 10000   #sul while    

tot_m = stats(means_sigma)

print(tot_m.mean())

fig, ax = plt.subplots ()
ax.scatter(x_ascisse, mean_value, s=2)
#ax.errorbar (x_ascisse, mean_value, xerr = 0.0, yerr = sigma_value)
#ax.plot (N_events, means_sigma, color = 'red', label = 'all toys')
#ax.plot (N_events, sigma_of_the_mean, color = 'blue', label = 'single toy')
ax.set_xscale ('log')
ax.legend ()
plt.show()

