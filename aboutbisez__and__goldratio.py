
#1:ALGORITMO DI BISEZIONE (funzione per trovare lo zero della funzione)

def bisezione (
    g,              # funzione di cui trovare lo zero
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione
    '''
    xAve = xMin 
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin) 
        if (g (xAve) * g (xMin) > 0.): xMin = xAve 
        else                         : xMax = xAve 
    return xAve 

#: versione ricorsiva
def bisezione_ricorsiva (
    g,              # funzione di cui trovare lo zero  
    xMin,           # minimo dell'intervallo            
    xMax,           # massimo dell'intervallo          
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione ricorsivo
    '''
    xAve = 0.5 * (xMax + xMin)
    if ((xMax - xMin) < prec): return xAve ;
    if (g (xAve) * g (xMin) > 0.): return bisezione_ricorsiva (g, xAve, xMax, prec) ;
    else                         : return bisezione_ricorsiva (g, xMin, xAve, prec) ;

def bisezione_mod (
    g,              # funzione di cui trovare lo zero
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo         
    prec = 0.0001): # precisione della funzione
    '''
    Funzione che calcola zeri
    con il metodo della bisezione,
    ritornando anche la collezione di intervalli considerati
    '''
    extremes = []
    xAve = xMin
    while ((xMax - xMin) > prec) :
        extremes.append ([xMin, xMax])
        xAve = 0.5 * (xMax + xMin) 
        if (g (xAve) * g (xMin) > 0.): xMin = xAve 
        else                         : xMax = xAve 
    return xAve , extremes

--> nel main: zero, extremes = bisezione_mod (func, xMin, xMax)
#------------
#2:SEZIONE AUREA(GR, trovare il minimo/max di una funzione)
#!/usr/bin/python

def sezioneAureaMin (
    g,              # funzione di cui trovare il punto critico
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) > g (x2)): 
            x0 = x3
            x1 = x1         
        else :
            x1 = x2
            x0 = x0          
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaMin_ricorsiva (
    g,              # funzione di cui trovare  il punto critico
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) > g (x2)) : return sezioneAureaMin_ricorsiva (g, x3, x1, prec)
    else                   : return sezioneAureaMin_ricorsiva (g, x0, x2, prec)   


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaMax (
    g,              # funzione di cui trovare  il punto critico
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) < g (x2)): 
            x0 = x3
            x1 = x1         
        else :
            x1 = x2
            x0 = x0          
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaMax_ricorsiva (
    g,              # funzione di cui trovare  il punto critico
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) < g (x2)) : return sezioneAureaMax_ricorsiva (g, x3, x1, prec)
    else                   : return sezioneAureaMax_ricorsiva (g, x0, x2, prec)   





#------------------------------------------------------
#------------------------------------------------------
#A: tempi di esecuzione algoritmo
import time

print ('Zero della funzione = ', bisezione (func, 0., 4.))

#tempo di esecuzione algortimo iterativo:
start=time.time()
a = bisezione (func, 0., 4.) #non metto il print qui, ma prima. altrimenti mi calcola anche il tempo della funzione print. quindi rinonimo con una variabile a e calcolo il tempo di esecuzione
end=time.time()
print(f" time bisezione = {1000.*(end-start):.3f}ms")

#analogamente
    start=time.time()
    bisezione_ricorsiva(func, 0., 4.) #se mettessi un print attorno a bisezione_... il tempo di esecuzione sarebbe piu lungo, per cui lo escludo e calcolo soltanto il valore
    end=time.time()
    execution_time=end-start
    print(f"time bisezione = {1000.*(end-start):.3f}ms")
    print(f"Il tempo di esecuzione della funzione tra i blocchi start ed end vale: {1000.*(execution_time):.3f}ms") #ok, f sta per troncaree alla terza cifra decimale del float
    print("il tempo di... vale: ", execution_time, "secondi") #in maniera piu rozza, dicono tutte e tre la stessa cosa


