##########################################################
##                                                      ##
## Module: dnf_1d.py                                    ##
##                                                      ##
## Version: 0.2                                         ##
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
## Checklist: 1- Enable multi bubbles                   ##
##            2- Enable path integration                ##
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
    c= 0.095            #Global inhibition
    nrate = 0.3       
    
    ###########################################
    
    #Takes a location between 0 and 2pi for both x along with the sigma
    # value for the gaussian function. Produces an narray with the gaussian
    # figure.   
    @staticmethod
    def gauss_pbc(locx,sig):
        w=zeros((dnf.n,1))
        for i in range(dnf.n):
                d=min([abs(i*dnf.dx-locx) , 2*pi-abs(i*dnf.dx-locx)]) 
                w[i]=1./(sqrt(2*pi)*sig)\
                    *exp(-(d**2/(2*sig**2)))
                    
 
                
        return w   
    
    @staticmethod
    def gauss_diff(locx,sig1,sig2):
        w=zeros((dnf.n,1))
        for i in range(dnf.n):
                d=min([abs(i*dnf.dx-locx) , 2*pi-abs(i*dnf.dx-locx)]) 
                w[i]=1./(sqrt(2*pi)*sig1)*exp(-(d**2/(2*sig1**2)))\
                    - 1./(sqrt(2*pi)*sig2)*exp(-(d**2/(2*sig2**2)))
        return w       
    
    
    #Initializor for the dynamic neural field. It creates a new one with the
    # parameters described above. In later versions it would support a multi
    # hypothesis dnf by supplying it with the parameter multi=True 
    def __init__(self,multi=False):
        self.t = 0
        self.gauss = dnf.gauss_pbc(pi,self.sig)
        self.u=zeros((dnf.n,1))      #The neural field state at a specific time
        if multi:
            self.c = 0.03
            self.z = 1000*(self.hebbMulti()-self.c)
        else:
            self.z = 1000*(self.hebb()-self.c)
            
        self.zpi1 = 1000*(self.hebbPI1()-self.c)
        self.zpi2 = 1000*(self.hebbPI2()-self.c)
        self.xall=[]
    
        
    #Produces the weight array that is responsible for integrating motion
    # to the right
    def hebbPI1(self):
        w=zeros((self.n,self.n))
        rbar = dnf.gauss_pbc(0,self.sig)
        for i in range(self.n):
            r=dnf.gauss_pbc(i*self.dx,self.sig)
            w=w+dot(r,rbar.transpose())
            rbar= (1-self.nrate)*rbar+ self.nrate*r
        return w/self.n
    
    #Produces the weight array that is responsible for integrating motion
    # to the left
    def hebbPI2(self):
        w=zeros((self.n,self.n))
        rbar = dnf.gauss_pbc(0,self.sig)
        for i in range(self.n):
            r=dnf.gauss_pbc((self.n-i)*self.dx,self.sig)
            w=w+dot(r,rbar.transpose())
            rbar= (1-self.nrate)*rbar+ self.nrate*r
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
    
    #Produces a weight array which supports multi hypotheses.
    def hebbMulti(self):
        w=zeros((self.n,self.n))   
        for i in range(self.n):
            r=dnf.gauss_pbc(i*self.dx,self.sig/3) - dnf.gauss_pbc(i*self.dx,self.sig)
            w=w+dot(r,r.transpose())
        return w/self.n
    
    
    #Takes a dynamic field state along with an activity input and the weight
    # from hebb. It updates the dynamic neural field and returns it after one
    # step in time. 
    def update(self,I, v=0):
        self.t += 1
        if v <= 0:
            zpi = self.zpi2
        else:
            zpi = self.zpi1
        r=0.5*(tanh(0.1*self.u)+1)
        z = (self.z + abs(v)*zpi)/(abs(v)+1)
        self.u=self.u+ self.tau*(-self.u+dot(z,r)*self.dx+I)
        x=0.5*(tanh(0.1*self.u)+1)
        self.xall = self.xall + [list(self.u.reshape(-1,))]
    
    
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
    dnfex = dnf(multi=True)
  
  
  
#    r = dnf.gauss_pbc(pi,dnf.sig/2) - dnf.gauss_pbc(pi,dnf.sig)
#    plt.plot(r)
#    plt.show()
  
    #Provide input at pi/2 for 50 steps
    I=dnf.gauss_pbc(pi,dnf.sig/10)
    for i in range(200):
       dnfex.update(I)
       
       
    I=zeros((dnf.n,1))
    for i in range(200):
        dnfex.update(I)   
        
    I=dnf.gauss_pbc(2*pi/3,dnf.sig/10)
    for i in range(300):
       dnfex.update(I)   

    I=zeros((dnf.n,1))
    for i in range(20):
        dnfex.update(I)   
        
        
    I=dnf.gauss_pbc(2*pi,dnf.sig/20)
    for i in range(200):
       dnfex.update(I)  

    I=zeros((dnf.n,1))
    for i in range(200):
        dnfex.update(I)  

    dnfex.plot()