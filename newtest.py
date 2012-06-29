"""
Last modified: 7 June 2012

"""


from numpy import *
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter



n=100
tau=0.1
u=zeros(n)
I=zeros(n)
dx=2*pi/n
sig=2*pi/10*0.6


def gauss_pbc(loc,sig):
    x=zeros(n)
    for i in range(n):
        d=min([abs(i*dx-loc), 2*pi-abs(i*dx-loc)])
        x[i]=1./(sqrt(2*pi)*sig)*exp(-d**2/(2*sig**2))
    return x
   
def hebb():
    w=zeros((n,n))
    for i in range(n):
        r=array([gauss_pbc(i*dx,sig)])
        w=w+dot(r.transpose(),r)
    return w/n
   
def update(u,I):
    x=0.5*(tanh(0.1*u)+1)
    u=u+tau*(-u+dot(w,x)*dx+I)
    return u


def hebbPI():
    w=zeros((n,n))
    for i in range(n):
        r=array([gauss_pbc(i*dx,sig)])
        r2=array([gauss_pbc(((i+5)%n)*dx,sig)])
        w=w+dot(r.transpose(),r2)
    return w/n


if __name__ == "__main__":
    #Initialize weights
    w=1000*(hebb()-0.095)
    #Initialize dnf state-history list    
    xall=[]
 

    I=zeros(n)
    for t in arange(20):
        u=update(u,I)
        x=0.5*(tanh(0.1*u)+1)
        xall=xall+[x]   
    #Provide input at pi/2 for 50 steps
#    I=gauss_pbc(pi/2,sig)
#    for t in arange(100):
#        u=update(u,I)
#        x=0.5*(tanh(0.1*u)+1)
#        xall=xall+[x]
#
#    #Wait till it stabilizes for 50 steps
#
#    
##    #Provide new input at 3pi/2 for 90 steps to overcome 
##    #the belief of the previous state
##    I=gauss_pbc(3 * pi / 2,sig)
##    for t in arange(90):
##        u=update(u,I)
##        x=0.5*(tanh(0.1*u)+1)
##        xall=xall+[x]
#
#    #Wait till it stabilizes for 50 steps
#    I=zeros(n)
#    for t in arange(200):
#        u=update(u,I)
#        x=0.5*(tanh(0.1*u)+1)
#        xall=xall+[x]
        
    #Create a figure for the 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = meshgrid(arange(n), arange(350))
    surf = ax.plot_surface(X, Y, asarray(xall), cmap=cm.jet,
        linewidth=0, antialiased=True)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_zlabel("Excitation")
    ax.set_xlabel("Node")
    ax.set_ylabel("Time")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    

    
    
    