##########################################################
##                                                      ##
## Module: dnf_1d.py                                    ##
##                                                      ##
## Version: 0.1                                         ##
##                                                      ##
## Description: A running simulation of a dynamic       ##
##  neural field with several modifications for         ##
##  improving runtime. This is the 1d version. It       ##
##  is used as a simple starting point for underst-     ##
##  ing path integration along with a working           ##
##  implementation of the mexican hat gaussians which   ##
##  enable multiple hypothesese on a single neural      ##
##  field                                               ##
##                                                      ##
## Checklist: 1- Enable multi bubbles          X        ##
##            2- Enable path integration       X        ##
##                                                      ##
##########################################################


from numpy import *
from scipy.signal import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class dnf:
    ######### Simulation variables ############
    n=100               #Number of nodes per side
    tau=0.1             #Tau
    I=zeros((n,1))      #Input activity
    dx=2*pi/n           #Length per node in the x dimension
    sig=2*pi/10*0.6     #Sigma of the gaussian function
    
    
    ###########################################
    
    #Takes a location between 0 and 2pi for both x and y along with the sigma
    # value for the gaussian function. Produces an n*n array with the gaussian
    # figure.
    @staticmethod
    def gauss_pbc(locx,sig):
        w=zeros((dnf.n,1))
        for i in range(dnf.n):
                d=min([abs(i*dnf.dx-locx) , 2*pi-abs(i*dnf.dx-locx)]) 
                w[i]=1./(sqrt(2*pi)*sig)\
                    *exp(-(d**2/(2*sig**2)))
                    
 
                
        return w   
    
    
    
    def __init__(self,multi=False):
        self.t = 0
        self.gauss = dnf.gauss_pbc(pi,self.sig)
        self.u=zeros((dnf.n,1))      #The neural field state at a specific time
        
        if multi:        
            self.z = 1000*(self.hebbMulti()-0.095)
        else:
            self.z = 1000*(self.hebb()-0.095)
            self.zpi1 = 1000*(self.hebbPI1()-0.095)
            self.zpi2 = 1000*(self.hebbPI2()-0.095)
        self.xall=[]
    
        
    
    def hebbPI1(self):
        w=zeros((self.n,self.n))
        for i in range(self.n):
            r=dnf.gauss_pbc(i*self.dx,self.sig)
            r2=dnf.gauss_pbc((i+5)%self.n*self.dx,self.sig)
            w=w+dot(r,r2.transpose())
        return w/self.n
    
    def hebbPI2(self):
        w=zeros((self.n,self.n))
        for i in range(self.n):
            r=dnf.gauss_pbc(i*self.dx,self.sig)
            r2=dnf.gauss_pbc((i-5)%self.n*self.dx,self.sig)
            w=w+dot(r,r2.transpose())
        return w/self.n
    
    #Uses hebbian learning to produce a single n*n array of the relation between
    # two gaussians. It yields the weights between the middle node and all the
    # others.
    def hebb(self):
        w=zeros((self.n,self.n))
        for i in range(self.n):
            r=dnf.gauss_pbc(i*self.dx,self.sig)
            w=w+dot(r,r.transpose())
        return w/self.n
    
    
    def hebbMulti(self):
        w=zeros((self.n,self.n))   
        for i in range(self.n):
            r=dnf.gauss_pbc(i*self.dx,0.2) - dnf.gauss_pbc(i*self.dx,dnf.sig)
            w=w+dot(r,r.transpose())
        return w/self.n
    
    
    #Takes a dynamic field state along with an activity input and the weight
    # from hebb. It updates the dynamic neural field and returns it after one
    # step in time. 
    def update(self,I, v):
        self.t += 1
        if v < 0:
            zpi = self.zpi2
        else:
            zpi = self.zpi1
        r=0.5*(tanh(0.1*self.u)+1)
        z = (self.z + abs(v)*zpi)/(abs(v)+1)
        self.u=self.u+ self.tau*(-self.u+dot(z,r)*self.dx+I)
        x=0.5*(tanh(0.1*self.u)+1)
        self.xall = self.xall + [list(x.reshape(-1,))]
    
    #Takes a dnf state and draws it in 3dwith respect to time.
    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.X, self.Y = meshgrid(
            arange(self.n),
            arange(self.t))      #A grid for plotting purposes        
        surf = ax.plot_surface(self.X, self.Y, self.xall, cmap=cm.jet,
            linewidth=0, antialiased=True)
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_zlabel("Excitation")
        ax.set_xlabel("Node 'X'")
        ax.set_ylabel("Time")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()




    
#Test case for the module
if __name__ == "__main__":
    dnfex = dnf()
  


  
    #Provide input at pi/2 for 50 steps
    I=dnf.gauss_pbc(pi,dnf.sig)
    for t in arange(300):
       dnfex.update(I,-1)
       
       
    I=zeros((dnf.n,1))
    for i in range(200):
        dnfex.update(I,-1)

        

    dnfex.plot()