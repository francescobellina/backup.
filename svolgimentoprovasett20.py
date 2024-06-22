
def logistic(b,x):
    return b*x*(1-x)



def recurrentLogistic(n,b,x):
   
   y=logistic(b,x)
   if n==1: return y
   else:
       return logistic(b,recurrentLogistic(n-1,b,x))

print(recurrentLogistic(3,1,0.5))

#main:
import numpy as np
import matplotlib.pyplot as plt

#mettere al posto di 6 un n da inserire in input
N=int(500)
stepa = np.linspace(0.1,3.9,N)
stepx= np.linspace(0.1,0.99,N)

y = []
for i in range(N):
    b=stepa[i]
    x=stepx[i]
    y.append(recurrentLogistic(100,b,x))
    
#print(y)

#plt.scatter(stepa,y, s=4)
#plt.axvline(x = 3.56995, color = 'red', ls='--', label = 'valore critico')
#plt.show()


N_=np.arange(1,101,1)
#print(N_)

a_=3
x0_=0.01

u=[]
for i in N_:
    u.append(recurrentLogistic(i,a_,x0_))

#print(u)
#plt.scatter(N_,u, s=4)
#plt.show()

fig, ax = plt.subplots(1,2)

ax[0].scatter(stepa,y, s=4, c='blue')
ax[0].axvline(x = 3.56995, color = 'red', ls='--', label = 'valore critico')
ax[1].scatter(N_,u, s= 5)
#ax.legend()
plt.show()