sample = np.loadtxt ('sample.txt')


#implementazione di base grafico
fig, ax = plt.subplots (nrows = 1, ncols = 1)

ax.plot (x_coord, y_coord_1, label='sin (x)')
ax.set_title ('Comparing trigonometric functions', size=14)
ax.set_xlabel ('x')
ax.set_ylabel ('y')
ax.legend ()

#hist
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample,
         color = 'orange',
        )

#bin control
bin_edges = np.linspace (xMin, xMax, N_bins)
print ('length of the bin_edges container:', len (bin_edges))
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample,
         bins = bin_edges,
         color = 'orange',
        )

#bin sturges
import numpy as np
def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

N_bins = sturges (len (sample))
bin_edges = np.linspace (xMin, xMax, N_bins)

#scala logaritmica
ax.set_yscale ('log')

#da scipy per la pdf normale
from scipy.stats import norm
mean = 1.
sigma = 0.5
x = mean + sigma / 2.
print (norm.pdf (x, mean, sigma))

#momenti della pdf
ave, var, skew, kurt = norm_fix.stats (moments='mvsk')
print (ave, var, skew, kurt)



#salvare la figura 
plt.savefig ('ex_3.1.png') #png

#utili:

plt.close() #è per i plt plots consecutivi
plt.xlabel('xlabel')
plt.ylabel('ylabel')

ax.set_ylim(bottom,up)
axes[0,0].set_ylim([0,40])

ax.set_yscale ('log')

histtype= 'step'

plt.axvline(x = 3.56995, color = 'red', ls='--', label = 'valore critico')

ax.set_title ('Plotting Functions in Matplotlib', size=14)

ax.set_xlim (-5, 5)
ax.set_ylim (-2, 2)

ax.plot ([means_stats.sigma (), means_stats.sigma ()], plt.ylim ()) #linea verticale ...

ax.errorbar (N_events, integrals, xerr = 0.0, yerr = integral_errors) #ERROR BAR ax.
ax.errorbar (x_coord, y_coord, xerr = 0.0, yerr = sigma_y, linestyle = 'None', marker = 'o') 

#FIT iminuit LS: display

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



#----------------------------------------
#ESEMPI


#1: confronto (due grafici affiancati)

data1 = np.genfromtxt('sample_data/eventi_unif.txt')
data2 = np.genfromtxt('sample_data/eventi_gauss.txt')

N_bins=int(max(len(data1),len(data2))/10) #usare in generale sturges:

N_bins_optimized=sturges(len(data1)) 

fig, axes = plt.subplots(2,2) #qui avrò una griglia 2x2 con quattro grafici

axes[0,0].hist(data1,
              bins=N_bins,
              label='eventi unif',
              color='blue'
              )
axes[0,0].set_ylim([0,40]) #cosi rendo confrontabili i grafici: il range prima andava da 0 a 20, 0-40 è il range di axes[0,1] (data2)

axes[0,1].hist(data2,
              bins=N_bins,
              label='eventi gauss',
              color='blue'
              )


axes[1,0].hist(data1,
              bins=N_bins_optimized,
              color='hotpink'
              )
axes[1,0].set_ylim([0,800])


axes[1,1].hist(data2,
              bins=N_bins_optimized,
              label='eventi gauss',
              color='hotpink'
              )





#2: confronto (sovrpposizione di istogrammi)
sample_1 = np.loadtxt ('\\Users\Francesco\Desktop\eventi_unif.txt')
sample_2 = np.loadtxt ('\\Users\Francesco\Desktop\eventi_gauss.txt')

xMin = floor (min (min (sample_1), min (sample_2)))
xMax = ceil (max (max (sample_1), max (sample_2)))
N_bins = sturges (min (len (sample_1), len (sample_2)))

bin_edges = np.linspace (xMin, xMax, N_bins)

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample_1,
             bins = bin_edges,
             color = 'orange',
            )
ax.hist (sample_2,
              bins = bin_edges,
              color = 'red',
          alpha = 0.5,
            )
#COSì PLOTTO I DUE ISTOGRAMMI IN UNO STESSO


ax.set_title ('Histogram example', size=14)
ax.set_xlabel ('variable')
ax.set_ylabel ('event counts per bin')
plt.show()


#------------------------------------------------
#------------------------------------------------
#A: UNIFORM PLOTTING DEL PIANO (X,Y)
#Mappatura uniforme del piano con numeri pseudo-casuali

import numpy as np
from math import floor
import matplotlib.pyplot as plt



def main () :


    xMin = float (0)
    xMax = float (10)
    yMin = float (50)
    yMax = float (-50)
    N_evt = int (100)
 
    x_coord = generate_range (xMin, xMax, N_evt)
    y_coord = generate_range (yMin, yMax, N_evt)

    fig, ax = plt.subplots ()
    ax.set_title ('uniform mapping', size=14)
    ax.set_xlabel ('x axis')
    ax.set_ylabel ('y axis')
    ax.scatter (x_coord, y_coord, s=4)
    plt.show ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":
    main ()
