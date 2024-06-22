
#1: punto preliminare, carico dati + istogramma iniziale
'''
1. Si disegni la distribuzione dei dati in un istogramma che abbia l’intervallo
di definizione scelto in funzione dei eventi raccolti ed il numero di bin che
permetta una agevole visualizzazione della distribuzione
'''

import numpy as np
import matplotlib.pyplot as plt
from math import ceil,floor

def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )



sample = np.loadtxt("\\Users\Francesco\Desktop\codes\misure_1.txt", delimiter='s', unpack=True)

xMin = min(sample)                 #operazioni di binning
xMax = max(sample)
N_bins = (sturges(len (sample)))
bin_edges = np.linspace (xMin, xMax, N_bins)

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample,
         bins = bin_edges,
         color = 'orange',
        )
plt.show()
plt.close()

#2: funzione LL sui dati
'''
2. Si costruisca una funzione in C++ che calcoli la verosimiglianza associata ai
dati, assumendo che gli eventi seguano una distribuzione esponenziale
'''
from math import exp, log

def exp_pdf (x, tau) :
    '''
    the exponential probability density function
    '''
    if tau == 0. : return 1.
    return exp (-1 * x / tau) / tau


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def likelihood (theta, pdf, sample) :
    '''
    the likelihood function calculated
    for a sample of independent variables idendically distributed 
    according to their pdf with parameter theta
    '''
    risultato = 1.
    for x in sample:
      risultato = risultato * pdf (x, theta)
    return risultato


# ---- ---- ---- ---- --

def loglikelihood (theta, pdf, sample) :
    '''
    the log-likelihood function calculated
    for a sample of independent variables idendically distributed 
    according to their pdf with parameter theta
    '''
    risultato = 0.
    for x in sample:
      if (pdf (x, theta) > 0.) : risultato = risultato + log (pdf (x, theta))  #condizione sul logaritmo (i valori devono essere positivi)   
    return risultato



#ho il sample (le x), non ho tau (sarà da determinare variando i k)
#per il momento fisso tau=1, (k-1)

tau = 1.

LL_func = likelihood(tau,exp_pdf,sample)
print("LL:", LL_func)

lLL_func = loglikelihood(tau,exp_pdf,sample)  #è un valore "puntuale", dal sample possiamo ricavare poi lo stimature per i parametri (in questo caso è uno solo) theta
print("lLL:", lLL_func)

#3: ora vario tau in base al sample e plotto l'andamento della funzione di verosomiglianza
'''
3. Si scriva un ciclo che disegni in un grafico l’andamento del valore
della verosimiglianza in funzione del valore del parametro k, dopo aver
scelto un ragionevole passo di campionamento dello stesso k, disegnando il TGraph in
un’immagine
'''
#sample  = sample[:int (len (sample)/10)]

taus = np.linspace(1.,3.,10)
sample2 = []
sample3 = []
for i in range(0,taus.size):
    t0=taus[i]
    sample2.append(likelihood(t0,exp_pdf,sample))
    sample3.append(loglikelihood(t0,exp_pdf,sample))


#plt.plot(taus,sample2)  #<-- opzionale, preferisco visualizzare la loglikelihood, eventualmente scommentare
#plt.show()
#plt.close()

plt.plot(taus,sample3)
plt.title("Andamento della LL in funzione di tau", size=8)
plt.xlabel('theta(tau)')
plt.ylabel('logLL')
plt.show()
plt.close()


#4: Maximum LL

'''
4. Si utilizzi il metodo della sezione aurea per deteminare il massimo della
verosimiglianza in un intervallo ragionevole di valori del parametro k, de-
terminato a partire dal plot disegnato
'''
#un set ragionevole di valori osservando il plot tracciato, come richiesto, potrebbe essere con tau (k^-1) compresi tra 0.5 e 3
#Essendoci in x=2 un punto di massimo locale (almeno graficamente) si preferisce includere quei valori per il momento.
#(si osservi il grafico sottostante):

plt.plot(taus,sample3)
plt.title("Andamento della LL in funzione di tau", size=8)
plt.xlabel('theta(tau)')
plt.ylabel('logLL')
plt.axvline(x = 1.5, color = 'red', ls='--')
plt.axvline(x = 2.5 , color = 'red', ls='--')
plt.show()
plt.close()

#Nella regione individuata è ragionevole aspettarsi il massimo della funzione (LL)

def sezioneAureaMax_LL (
    g,              # funzione di likelihood trovare il massimo
    pdf,            # probability density function of the events
    sample,         # sample of the events
    x0,             # estremo dell'intervallo  -->(tau min)        
    x1,             # altro estremo dell'intervallo -->   (tau max)     
    prec = 0.0001): # precisione della funzione   

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g(x3,pdf,sample) < g(x2,pdf,sample)): 
            x0 = x3
            x1 = x1         
        else :
            x1 = x2
            x0 = x0          
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 


tau_hat = sezioneAureaMax_LL (loglikelihood, exp_pdf, sample, 0.5, 3.)
print ('il valore di tau che massimizza il logaritmo della verosimiglianza è:', tau_hat)

plt.plot(taus,sample3)
plt.title("Andamento della LL in funzione di tau", size=8)
plt.xlabel('theta(tau)')
plt.ylabel('logLL')
plt.axvline(x = 1.5, color = 'red', ls='--')
plt.axvline(x = 2.5, color = 'red', ls='--')

x_hat= tau_hat
y_hat= loglikelihood(tau_hat,exp_pdf,sample)
plt.scatter(x_hat,y_hat, color="green")
plt.show()
plt.close()


#5: Narrowing
'''
5. Si mostri che il profilo della verosimiglianza in funzione del parametro k
diventa più largo al diminuire del numero di eventi a disposizione
--> equivalentemente: si mostri che il profilo della LL in funzione di theta (tau) diventa più stretto 
   all'aumentare del numero di eventi a disposizione
'''


sample_size = 5
N_evt = len(sample)
fig, ax = plt.subplots ()
ax.set_title ('Log-likelihood ratio scan', size=14)
ax.set_xlabel ('tau')
ax.set_ylabel ('LLR')
tauaxis = np.linspace (0.1, 1.5, 1000)
colors = [
          'thistle',
          'blue',
          'hotpink',
          'red',
         ]
index = 0
#fig, axes = plt.subplots(2,2) 
while sample_size <= N_evt :
    # create a subsample
    subsample = sample[:int (sample_size)]
    llr = np.arange (0., tauaxis.size)
    for i in range (tauaxis.size) :
        tau_hat = sezioneAureaMax_LL (loglikelihood, exp_pdf, subsample, 1.5, 3., 0.0001)    
        llr[i] = loglikelihood (tauaxis[i], exp_pdf, subsample)
    plt.plot (tauaxis, llr, color = colors[index], label = str (sample_size) + ' events')
    #axes[0,0].plot(tauaxis, llr)
    #axes[1,0].plot(tauaxis, llr)
    #axes[0,1].plot(tauaxis, llr)
    #axes[1,1].plot(tauaxis, llr)

    index = index + 1
    sample_size = sample_size + 5

plt.legend ()
plt.show ()

