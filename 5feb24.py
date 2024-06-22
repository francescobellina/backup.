#!/usr/bin/python

import random
from math import sqrt
from math import floor
import numpy as np
from iminuit import Minuit             
from iminuit.cost import LeastSquares 
import matplotlib.pyplot as plt
from stats import stats

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

#---------------------------------------------------


def parabola (x, a, b, c) :
    '''
    reference model to be fitted
    '''
    return a + b * x + c * x * x

# initial parameters of the problem

a_true = 3.
b_true = 2.
c_true = 1.

'''1: Si definisca una funzione ϕ(x, a, b, c) che traccia un andamento parabolico in funzione di x e se ne
disegni disegni l’andamento nell’intervallo (0, 10) ...'''

x_coord = np.linspace (0,10, 10000)
y_coord = parabola (x_coord,a_true,b_true,c_true)

fig,ax = plt.subplots(1,1)
ax.plot(x_coord,y_coord)
plt.title("andamento nell'intervallo (0,10) della funzione parabola")
plt.show()
plt.close()
#--------------------

'''
2: Si generino N = 10 punti xi distribuiti in modo pseudo-casuale secondo una distribuzione uniforme
sull’intervallo orizziontale e si associ a ciascuno di essi una coordinata
yi = ϕ(xi, a, b, c) + εi,
dove εi è un numero pseudo casuale generato, con il metodo del teorema centrale del limite, secondo
una distribuzione Gaussiana di media 0 e deviazione standard σy = 10.
'''
xx_coord_unif = generate_range(0.,10.,10)
xx_coord_unif = np.random.uniform(0.,10.,10)

epsilons = generate_TCL_ms (0.,10.,10)
epsilon_sigma = 10. #<--- è la deviazione standard (sulle y) che ho utilizzando nella riga precedente
y_coord_eps = []
for i in range(len(xx_coord_unif)):
    y_coord_eps.append(parabola(xx_coord_unif[i],a_true,b_true,c_true)+epsilons[i]) #costruzione degli yi come da tra
    
fig,ax = plt.subplots(1,1)
ax.scatter(xx_coord_unif,y_coord_eps, s=4)
ax.errorbar (xx_coord_unif,y_coord_eps, xerr = 0.0, yerr = epsilon_sigma, linestyle = 'None', marker = 'o') 
plt.show()
plt.close()

'''
3: Si faccia un fit della funzione ϕ(x, a, b, c) sul campione così generato (che tecnica bisogna utiliz-
zare?)

Si procede ora all'esecuzione del fit sul campione mediante la tecnica dei minimi quadrati, utilizzando il modulo
iminuit.cost invocando la funzione LeastSquare della libreria iminuit procedendo alle operazioni di fit
'''
from iminuit import Minuit             
from iminuit.cost import LeastSquares 

y_error= np.ones(len(y_coord_eps))*epsilon_sigma
least_squares = LeastSquares(xx_coord_unif,y_coord_eps,y_error, parabola) #GENERO LA LS FUNZIONE; sigma_y è fissata
#le applico:
my_minuit = Minuit (least_squares, a=0., b=0., c=0.)  # starting values for m and q <------ 
my_minuit.migrad () # finds minimum of least_squares function
my_minuit.hesse ()   # accurately computes uncertainties
print(my_minuit.fval)

is_valid = my_minuit.valid
print ('success of the fit: ', is_valid)

# draw data and fitted line
xx=np.linspace(0,10,100)
plt.errorbar(xx_coord_unif, y_coord_eps, y_error, fmt="ok", label="data")
plt.plot(xx, parabola(xx, *my_minuit.values), label="fit", color="red")

# display legend with some fit info
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {my_minuit.fval:.1f} / {my_minuit.ndof:.0f} = {my_minuit.fmin.reduced_chi2:.1f}",
]
for p, v, e in zip(my_minuit.parameters, my_minuit.values, my_minuit.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel("x")
plt.ylabel("y");
plt.show()
plt.close()

'''
4: Si costruisca la distribuzione del Q2 a partire dal fit effettuato, ripetendolo molte volte utilizzando
toy experiment

'''


N_toys = 10000
Q_squares = []
xx_unif_toy = np.arange (0, 10, 1)
y_eps_toy = np.zeros (10)
epsilon_sigma = 10

for i_toy in range (N_toys) :
    epsilons_toy = generate_TCL_ms (0., epsilon_sigma, 10)
    for i in range (xx_unif_toy.size) :
        y_eps_toy[i] = parabola (xx_unif_toy[i], a_true, b_true,c_true) + epsilons_toy[i]    #è sempre uguale ma le espilons_toy hanno fluttuazioni rand
    least_squares = LeastSquares(xx_unif_toy,y_eps_toy,y_error, parabola)  #FUNZIONE LEASTSQUARES 
    my_minuit_toy = Minuit (least_squares, a = 0, b = 0, c=0)  # starting values for m and q
    my_minuit_toy.migrad ()  # finds minimum of least_squares function
    my_minuit_toy.hesse ()   # accurately computes uncertainties
    if my_minuit_toy.valid : 
        Q_squares.append (my_minuit_toy.fval)   #COSTRUZIONE DELLA DISTRIBUZIONE Q2



'''
5:Si svolgano i punti precedenti generando gli scarti εi secondo una distribuzione uniforme che ab-
bia la stessa deviazione standard della Gaussiana, disegnando poi la distribuzione del Q2 così ot-
tenuto sovrapposta a quella precedente (per una visualizzazione migliore, si può utilizzare l’opzione
histtype=’step’).
'''

N_toys = 10000
Q_squares_2 = []
xx_unif_toy = np.arange (0, 10, 1)
y_eps_toy = np.zeros (10)
epsilon_sigma = 10
#y_error= np.ones(len(y_coord_eps))*epsilon_sigma

for i_toy in range (N_toys) :
    epsilons_toy = generate_TCL_ms (0., epsilon_sigma, 10)
    epsilons_toy_2= np.random.uniform (-epsilon_sigma*1.732 , epsilon_sigma*1.732 ,10)
    for i in range (xx_unif_toy.size) :
        y_eps_toy[i] = parabola (xx_unif_toy[i], a_true, b_true,c_true) + epsilons_toy_2[i]    #è sempre uguale ma le espilons_toy hanno fluttuazioni rand
    least_squares = LeastSquares(xx_unif_toy,y_eps_toy,y_error, parabola)  #FUNZIONE LEASTSQUARES 
    my_minuit_toy = Minuit (least_squares, a = 0, b = 0, c=0)  # starting values for m and q
    my_minuit_toy.migrad ()  # finds minimum of least_squares function
    my_minuit_toy.hesse ()   # accurately computes uncertainties
    if my_minuit_toy.valid : 
        Q_squares_2.append (my_minuit_toy.fval)   #COSTRUZIONE DELLA DISTRIBUZIONE Q2

#Plotting:
fig, ax = plt.subplots ()
ax.set_title ('Q-squared distribution', size=14)
ax.set_xlabel('q_squared')
ax.set_ylabel('events in bin')
N_dof = my_minuit.ndof
bin_edges = np.linspace (0, 4 * N_dof, floor (N_toys/100))   # edges o the histogram bins
ax.hist (Q_squares,
         bins = bin_edges,
         color = 'orange',
        )
ax.hist (Q_squares_2,
         bins = bin_edges,
         color = 'blue',
         histtype="step",
        )
ax.set_title ('Q2 distributions', size=14)
ax.set_xlabel ('Q2')
ax.set_ylabel ('event counts per bin')
ax.legend ()
plt.show ()


'''
6:In funzione della distribuzione ottenuta per il Q2, si determini la soglia oltre la quale rigettare il
risultato del fit, dato il suo valore di Q2, per ottenere un p-value maggiore o uguale di 0.10
'''
# ordina i valori di Q2
# scorri i valori fino a che la frazione è maggiore di 0.9... cioè prendi l'elemento che sta all'indice 90% del totale
from math import floor
N_soglia = floor (N_toys * 0.9)
Q_squares_2.sort ()  #ordine crescente 
val_soglia = Q_squares_2[N_soglia] #valore di soglia oltre al quale rigettare?
print ('soglia al 90%:', ) #stampa, una volta ordinati i Q2, il valore a circa 90% del campione

Q2_squares_2_rigettati = []
for val in Q_squares_2:
    if val > val_soglia: Q2_squares_2_rigettati.append(val)


fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (Q_squares_2,
         bins = bin_edges,
         color = 'blue',
         label = 'test statistics',
         histtype='step',
        )
ax.hist (Q2_squares_2_rigettati,
         bins = bin_edges,
         color = 'lightblue',
         label = 'rigettati',
        )
ax.set_title ('Q2 distributions', size=14)
ax.set_xlabel ('Q2')
ax.set_ylabel ('event counts per bin')
ax.legend ()
plt.show ()

