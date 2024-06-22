
#Write a Python program to draw a Gaussian distribution and its cumulative function
#Write a Python program to draw an exponential distribution and its cumulative function
#Write a Python program to draw a binomial distribution and its cumulative function
#Write a Python program to draw a Poisson distribution for several values of its mean, overlapped

from scipy.stats import norm
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
#--
def Gaussian (x, mean, sigma) :   #FUNZIONE GAUSSIANA
    if sigma == 0 : return float (x == 0.)
    return np.exp (-0.5 * pow ((x - mean) / sigma, 2.)) / (sqrt (2 * np.pi) * sigma) #forma analitica

#-
mean = 1.
sigma = 0.5
x = mean + sigma / 2.
norm.pdf (x, mean, sigma)

#--

#fig, ax = plt.subplots(1,2)
normal = norm (100., 10.) #I DUE VALORI SONO LA MEDIA E LA SIGMA (che di default sono 0 e 1)

x_axis = np.linspace (50., 150., 100) #input sulle x
plt.plot(x_axis, normal.pdf (x_axis), label="PDF")
#plt.plot (x_axis, normal.pdf (x_axis), label="PDF") #funzione diiistribuzione normale
plt.legend ()
plt.savefig ('ex_3.7_pdf.png')
plt.show()

plt.clf () # clear the figure COSI ME NE PLOTTA DUE CONSECUTIVE

plt.plot (x_axis, normal.cdf (x_axis), color="red",label="CDF")
plt.legend ()
plt.show()
#plt.savefig ('ex_3.7_cdf.png')