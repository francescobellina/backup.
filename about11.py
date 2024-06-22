import myrand
from myrand import generate_TCL_ms
from matplotlib import pyplot as plt
import numpy as np

def func(x, m, q):
    '''
    referen per il modello da fittare
    '''
    return x*m + q


m_true = 0.5
q_true = 1.1
epsilon_sigma = 0.3


epsilons = generate_TCL_ms (0., epsilon_sigma, 10)

x_coord = np.arange (0, 10, 1)
y_coord = np.zeros (10)


for i in range (x_coord.size) :
    y_coord[i] = func (x_coord[i], m_true, q_true) + epsilons[i]

print(y_coord)
#questo for poteva essere omesso lavorando direttamente tra vettori numpy_
y_coord = func(x_coord, m_true, q_true) + epsilons
#sono della stessa dim lo posso fare

sigma_y = epsilon_sigma * np.ones((len(y_coord))) #idem è implicito un ciclo for grazie alla
#così la funzione si dice VETTORIZZATA
print(sigma_y)

#-------------------------------------------
#avrei potuto fare lo stesso così:
sigma_y = np.zeros(10)
for i in range(len(y_coord)):
    sigma_y[i] = epsilon_sigma
print(sigma_y)
#va da se che l'approccio cui sopra ottimizza il codice.
#------------------------------------------------


fig, ax = plt.subplots()
ax.set_title('linear model', size = 14)
ax.set_xlabel ('x')
ax.set_ylabel ('y')
ax.errorbar(x_coord, y_coord, xerr = 0.0, yerr = sigma_y, linestyle = 'None', marker = 'o') #con le y con tutte lo stesso sigma
#ax.errorbar al posto di ax.plot; da TENERE IN CONSIDERAZIONE  ho il grafico degli errori sui punti
plt.show()




#---------------------------
# VEDIAMO LA LIBRERIA iminuit (c'è anche alternativamente scipy ecc)
#iminuit è molto semplice da applicare
from iminuit import Minuit
from iminuit.cost import LeastSquares #cost function è praticamente la Q^2


# generate a least-squares cost function
least_squares = LeastSquares (x_coord, y_coord, sigma_y, func)  #somma(xi-phi(yi)/sigma)^2
my_minuit = Minuit (least_squares, m = 0, q = 0)  # starting values for m and q. PARAMETRI PER LA MINIMIZZAZIONE
#DELLA Q^2 con m e q parametri iniziali del fit 
my_minuit.migrad ()  # finds minimum of least_squares function (MINIMIZZO LA FUNZIONE)
my_minuit.hesse ()   # accurately computes uncertainties (STIMA DELL'ERRORE ASSOCIATO AI PARAMETRI, tramite le derivate seconde)
#hesse sfrutta una approssimazione della matrice delle covarianze



#ANALISI DEL FIT: convergenza

  # global characteristics of the fit
is_valid = my_minuit.valid
print ('success of the fit: ', is_valid)  #se converge o meno, imposto una precisione del minimo (è un metodo numerico, può non funzionare)


Q_squared = my_minuit.fval
print ('value of the fit Q-squared', Q_squared) #valore del Q^2
N_dof = my_minuit.ndof
print ('value of the number of degrees of freedom', N_dof) #numero dei gradi di libertà

#ANALISI DEL FIT : QUALITà: chi-2 e p-value

#The p-value associated to the fit result may be calculated from the cumulative distribution function of the
#probability density function:
from scipy.stats import chi2
# ...
print ('associated p-value: ', 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof))
#CALCOLO IL p-value PARTENDO DA chi2 e fval e ndof

#RISULTATI DEI FIT: parametri e incertezze

for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
    print(f'{par} = {val:.3f} +/- {err:.3f}') # formatted output
    
m_fit = my_minuit.values[0]
q_fit = my_minuit.values[1]  

#parameters values ed errors sono tutti parametri di python stores in appositi array

print (my_minuit.covariance) #posso stampare direttamente la MATRICE DI COVARIANZA

print (my_minuit.covariance[0][1])
print (my_minuit.covariance['m']['q'])    #Così accedo a par, val, err attraverso la matrice di covarianza

print (my_minuit.covariance.correlation ()) #MATRICE DI CORRELAZIONE (coefficienti di correlazione sui parametri)


display (my_minuit)