#!/usr/bin/python

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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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

#---------------------------------------------------

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


#---------------------------------------------------

def intersect_LLR (
    g,              # funzione di cui trovare lo zero
    pdf,            # probability density function of the events
    sample,         # sample of the events
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo 
    ylevel,         # value of the horizontal intersection    
    theta_hat,      # maximum of the likelihood    
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione
    '''
    def gprime (x) :
        return g (x, pdf, sample, theta_hat) - ylevel   #<------------------!!!!

    xAve = xMin 
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin) 
        if (gprime (xAve) * gprime (xMin) > 0.) : xMin = xAve 
        else                                    : xMax = xAve 
    return xAve 

#tau_hat = sezioneAureaMax_LL (loglikelihood, exp_pdf, sample, 0.5, 5., 0.0001)    
#tau_hat_minusS = intersect_LLR (loglikelihood, exp_pdf, sample, 0.5, tau_hat, -0.5, tau_hat)
#tau_hat_plusS = intersect_LLR (loglikelihood, exp_pdf, sample, tau_hat, 5., -0.5, tau_hat)
