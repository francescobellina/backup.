import numpy as np
from math import cos,pi
from math import sqrt
import random
#from myrand import generate_range, rand_range

###################


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

##################

def my_pdf(x):
    f=np.cos(x)*np.cos(x)
    a=0.424413182
    if x<=0 or  x>1.5*pi:
        return 0
    else: 
        return  a*f

'''
def f_norm(x,a):
    f=cos(x)*cos(x)
    
    if x<=0 or  x>1.5*pi:
        return 0
    else: 
        return  a*f 

def integral_HOM (func,a, xMin, xMax, yMax, N_evt) :


    x_coord = generate_range (xMin, xMax, N_evt)
    y_coord = generate_range (0., yMax, N_evt)

    points_under = 0
    for x, y in zip (x_coord, y_coord):
        if (func (x,a) > y) : points_under = points_under + 1 

    A_rett = (xMax - xMin) * yMax
    frac = float (points_under) / float (N_evt)
    integral = A_rett * frac
    integral_unc = A_rett**2 * frac * (1 - frac) / N_evt #incertezza secondo la varianza derivante dalla natura binomiale di nhit (points_under)
    return integral

a_Norms = np.random.uniform(0.,1.,100) #set di costanti di normalizzazione, trovare quella che minimizza f_norm
#print(a_Norms)
#estremi di integrazione:
xMin = 0
xMax = 1.5*pi
yMax= 1

x=[]

for i in range(len(a_Norms)):
    x.append(integral_HOM(f_norm,a_Norms[i],xMin,xMax,yMax,100))
'''    

#2:Si generi un insieme di 10000 numeri pseudo-casuali xi distribuiti secondo la pdf f (x) utilizzando
#il metodo try-and-catch.

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


xMin = 0
xMax = 1.5*pi
yMax= 1
N_ev=10000
sample = generate_TAC(my_pdf,xMin, xMax, yMax, N_ev)

#3:Si mostri in un istogramma la distribuzione degli eventi generati
import matplotlib.pyplot as plt

def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

N_bins = sturges (len (sample))

bin_edges = np.linspace (xMin, xMax, N_bins)

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample,
         bins = bin_edges,
         color = 'orange',
        )
#plt.axvline(x = pi*0.5, color = 'cyan', ls='--', label = 'valore critico')
plt.show()
plt.close()

#4:Si calcolino, a partire dagli eventi generati, la media, deviazione standard, asimmetria e curtosi della
#distribuzione implementando le funzioni corrispondenti.
from stats import stats

my_stats = stats(sample)

print("La media della distribuzione vale: ", my_stats.mean())
print("La deviazione standard della distribuzione vale: ", my_stats.sigma())
print("La assimetria (skewness) della distribuzione vale: ", my_stats.skewness())
print("La curtosi della distribuzione vale: ", my_stats.kurtosis())

#5:Si mostri quantitativamente che il teorema centrale del limite vale in questo caso, a partire dalla
#generazione di numeri pseudo-casuali con la tecnica del teorema centrale del limite applicata alla
#distribuzione f (x)

#dobbiamo mostrare con la TCL (su f) che essa e i suoi parametri assumono, in regime asintotico, una forma e carattere gaussiano 

def rand_TAC_pdf (f, xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    yMax=1
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


def rand_TCL_pdf (f,xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_TAC_pdf (f,xMin, xMax)
    y /= N_sum ;
    return y ;


def generate_TCL (f,xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL_pdf(f,xMin, xMax, N_sum))
    return randlist
 

#100,100,1000,10000
N_evt=10
N_max=1000
X = []


sample_gauss= generate_TCL(my_pdf,xMin, xMax, 10000, N_sum = 10, seed = 0.)
#print(sample_gauss)

N_bins = sturges(len (sample_gauss))
bin_edges = np.linspace (0, 1.5 * np.pi, N_bins)



fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample_gauss,
         bins = bin_edges,
         color = 'blue',
        )
#plt.axvline(x = pi*0.5, color = 'cyan', ls='--', label = 'valore critico')
plt.show()
plt.close()











#####################################
#####################################



import numpy as np
import matplotlib.pyplot as plt
from lib import pdf, genera, sturges, rand_TCL
from stats import stats


if __name__ == '__main__':

  # draw the pdf
  # ---- ---- ---- ---- ---- ---- ---- 

  fig, ax = plt.subplots (nrows = 1, ncols = 1)

  # preparing the set of points to be drawn 
  x_coord = np.linspace (0, 1.5 * np.pi, 10000)
  y_coord_1 = pdf (x_coord)

  # visualisation of the image
  ax.plot (x_coord, y_coord_1, label='pdf')
  ax.set_title ('probability density function', size=14)
  ax.set_xlabel ('x')
  ax.set_ylabel ('y')
  plt.savefig ('pdf.png')
#  plt.show ()

  # generate the sample and calculate the integral
  # ---- ---- ---- ---- ---- ---- ---- 

  campione, area = genera (10000)

  # draw the histogram of the sample
  # ---- ---- ---- ---- ---- ---- ---- 

  ax.set_title ('generated sample', size=14)
  print ('generati', len (campione),'eventi')
  print ("l'area della pdf prima della normalizzazione è",area)
  print ('il fattore di normalizzazione è', 1./area)
  N_bins = sturges (len (campione))
  bin_edges = np.linspace (0, 1.5 * np.pi, N_bins)
  ax.hist (campione,
           bin_edges,
           color = 'orange',
          )
  plt.savefig ('histo.png')

  # calculate moments
  # ---- ---- ---- ---- ---- ---- ---- 

  my_stats = stats (campione)
  print ('mean    :', my_stats.mean ())
  print ('sigma   :', my_stats.sigma ())
  print ('skewness:', my_stats.skewness ())
  print ('kurtosis:', my_stats.kurtosis ())

  # study the Gaussian behaviour
  # ---- ---- ---- ---- ---- ---- ---- 

  N_events = 10000
  means = []
  sigmas = []
  skews = []
  kurts = []
  x_axis = [2**j for j in range(0,6)]
  for N_sum in x_axis:
    campione_loc = [rand_TCL (N_sum) for j in range (N_events)]
    my_stats = stats (campione_loc)
    means.append (my_stats.mean ())
    sigmas.append (my_stats.sigma ())
    skews.append (my_stats.skewness ())
    kurts.append (my_stats.kurtosis ())

  fig, ax = plt.subplots (nrows = 4, ncols = 1)
  ax[0].plot (x_axis, means, label='mean')
  ax[1].plot (x_axis, sigmas, label='sigma')
  ax[2].plot (x_axis, skews, label='skewness')
  ax[3].plot (x_axis, kurts, label='kurtosis')
  plt.savefig ('stats.png')

  campione_gaus = [rand_TCL (32) for j in range (N_events)]

  fig, ax = plt.subplots (nrows = 1, ncols = 1)
  N_bins = sturges (len (campione_gaus))
  bin_edges = np.linspace (0, 1.5 * np.pi, N_bins)
  bin_content, _, _ = ax.hist (campione_gaus,
           bin_edges,
           color = 'orange',
          )
  plt.savefig ('gauss.png')

  from iminuit import Minuit
  from iminuit.cost import BinnedNLL
  from scipy.stats import norm
  from lib import mod_gaus

  my_stats_gaus = stats (campione_gaus)

  # the cost function for the fit
#  my_cost_func = BinnedNLL (bin_content, bin_edges, gaus_model)
  my_cost_func = BinnedNLL (bin_content, bin_edges, mod_gaus)

  my_minuit = Minuit (my_cost_func, 
                      mu = my_stats_gaus.mean (), 
                      sigma = my_stats_gaus.sigma ())

  my_minuit.migrad ()
  my_minuit.minos ()
  print (my_minuit.valid)
  from scipy.stats import chi2
  print ('associated p-value: ', 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof))
  if 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof) > 0.10:
    print ('the event sample is compatible with a Gaussian distribution')