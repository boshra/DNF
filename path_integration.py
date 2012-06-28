##########################################################
##                                                      ##
## Module: path_integration.py                                    ##
##                                                      ##
## Version: 0.1                                         ##
##                                                      ##
## Description: A running simulation of a dynamic       ##
##  neural field with several modifications for         ##
##  improving runtime.                                  ##
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
    u=zeros((n,n))      #The neural field state at a specific time
    I=zeros((n,n))      #Input activity
    dx=2*pi/n           #Length per node in the x dimension
    dy=2*pi/n           #Length per node in the y dimension
    sig=2*pi/11*0.6     #Sigma of the gaussian function
    X, Y = meshgrid(
        arange(n),
        arange(n))      #A grid for plotting purposes
    
    ###########################################
    #Takes a location between 0 and 2pi for both x and y along with the sigma
    # value for the gaussian function. Produces an n*n array with the gaussian
    # figure.
    @staticmethod
    def gauss_pbc(locx,locy,sig):
        z=zeros((dnf.n,dnf.n))
        for i in range(dnf.n):
            for j in range(dnf.n):
                d=min([abs(i*dnf.dx-locx) , 2*pi-abs(i*dnf.dx-locx)]) 
                d2=min([abs(j*dnf.dy-locy), 2*pi-abs(j*dnf.dy-locy)]) 
                z[j][i]=1./(sqrt(2*pi)*sig)\
                    *exp(-(d**2/(2*sig**2)+(d2**2/(2*sig**2))))
        return z    
    
    
    
    def __init__(self):
        self.gauss = dnf.gauss_pbc(pi,pi,self.sig)
        self.gauss2 = dnf.gauss_pbc(2*pi,pi,self.sig)
        self.z = 1000*(self.hebb()-0.095)
        
    
    
    
    
    #Uses hebbian learning to produce a single n*n array of the relation between
    # two gaussians. It yields the weights between the middle node and all the
    # others.
    def hebb(self):
        z = convolve2d(self.gauss, self.gauss2, 'same', 'wrap')
        return z/(self.n*2)
    
    #Takes a dynamic field state along with an activity input and the weight
    # from hebb. It updates the dynamic neural field and returns it after one
    # step in time. 
    def update(self,I):
        r=0.5*(tanh(0.1*self.u)+1)
        convo = convolve2d(r,self.z,'same','wrap')
        self.u=self.u+self.tau*(-self.u+convo*self.dx+I)
    
    
    #Takes a dnf state and draws it in 3d. 
    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.X, self.Y, self.u, cmap=cm.jet,
            linewidth=0, antialiased=True)
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_zlabel("Excitation")
        ax.set_xlabel("Node 'X'")
        ax.set_ylabel("Node 'Y'")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


#Test case for the module
if __name__ == "__main__":
    dnfex = dnf()
 
    I=zeros((dnf.n,dnf.n))
    for t in arange(20):
        dnfex.update(I)
        
    dnfex.plot()

   
    #Provide input at pi/2 for 50 steps
    I=dnf.gauss_pbc(3*pi/2,3*pi/2,dnf.sig)
    for t in arange(50):
       dnfex.update(I)

    dnfex.plot()



    d
#    I=dnf.gauss_pbc(pi/2,pi/2,dnf.sig)
#    for t in arange(10):
#        dnfex.update(I)