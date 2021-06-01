# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 04:15:18 2021

@author: juan felipe patino and aritro gosh
"""

from matplotlib import pyplot as plt
import time
import numpy as np
import matplotlib.animation as animation

m=1
nodes= int(128*m) #number of nodes
print("nodes=",nodes)
a = 0.01 #grediant energy coeficient
dt= 1e-6 #time steping
dx=1/(nodes-1) 
t=0.012 #final time
N_it=int(t/dt) #number of iterations

#define the matrices 
C_old=np.zeros((nodes,nodes))
C_new=np.zeros((nodes,nodes))
error=np.zeros((nodes,nodes))
F=np.zeros((nodes,nodes))

C_old=np.random.uniform(-1,1,(nodes,nodes))

#boundary condition 
C_old[0,:] = C_old[nodes-2,:]
C_old[nodes-1,:] = C_old[1,:]
C_old[:,0] = C_old[:,nodes-2]
C_old[:,nodes-1] = C_old[:,1]

# Declaration of movie writer
file = animation.FFMpegWriter(fps=5)

fig ,(ax1,ax2)= plt.subplots(1,2,figsize=(14,8))

def saver(C_n,e):
    "Error definition"
    errormax = np.max(e)
    "ploting evolution"
    im1 = ax1.imshow((C_n), vmin = -1, vmax = 1, cmap="brg") 
    ax1.invert_yaxis()
    cb1 = fig.colorbar(im1, ax=ax1,orientation="horizontal", anchor=(0, 0.01), shrink=0.6, pad=0.05)
    cb1.set_label('Species concentration')
    ax1.set_title("Concentration Evolution",loc='center',fontsize=(15))
    title1=ax1.text(40*m, 140*m, "",fontsize=(20),weight="bold")
    title1.set_text("Cahn-Hilliard Central difference ($\Delta$t={:.1e} s)".format(dt))
    title2 = ax1.text(.5, .1, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax1.transAxes, ha="center",weight="bold")
    title2.set_text("Time= {:.1e}".format(t_ac))
    "ploting error"
    im2 = ax2.imshow(e*100,vmin = 0, vmax = "{:5e}".format(errormax*100), cmap="brg") 
    ax2.invert_yaxis()
    cb2 = fig.colorbar(im2, ax=ax2, orientation="horizontal",anchor=(0, 0.01), shrink=0.6, pad=0.05)
    cb2.set_label('Error %')
    ax2.set_title("Error Evolution",loc='center',fontsize=(15))
    title3 = ax2.text(.5, .1, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax2.transAxes, ha="center",weight="bold")
    title3.set_text("Max Error= {:.2f}%".format(errormax*100))
    title4=ax1.text(120*m, -20*m, "",bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},color="b",fontsize=(13),weight="bold")
    title4.set_text("CPU_time={:.1f} s".format(RT))
    plt.show()
    file.grab_frame()
    print("time=",t_ac) 
    cb1.remove()
    cb2.remove()
    ax1.clear()
    ax2.clear() 

def cdf(C_o):
    for i in range (1,nodes-1):
            for j in range (1,nodes-1):
                F[i,j] = C_o[i,j]**3 - C_o[i,j] - a**2* ( C_o[i-1,j] + C_o[i+1,j] + C_o[i,j-1] + C_o[i,j+1] - 4*C_o[i,j] )/(dx**2)               
                C_new[i,j] = C_o[i,j] + dt*( F[i-1,j] + F[i+1,j] + F[i,j-1] + F[i,j+1] - 4*F[i,j] )/(dx**2)      
                error[i,j]=abs(dt*( F[i-1,j] + F[i+1,j] + F[i,j-1] + F[i,j+1] - 4*F[i,j] )/(dx**2))
    return C_new, error

with  file.saving(fig, "Cahn_Hilliard EUD nodes=128 dt=1e-6.mp4", 100):
    start=time.time()
    for n in range (0,N_it):
        t_ac = n*dt
        if n>=0:
            C_new, error=cdf(C_old)
 
        C_new[0,:] = C_new[nodes-2,:]
        C_new[nodes-1,:] = C_new[1,:]
        C_new[:,0] = C_old[:,nodes-2]
        C_new[:,nodes-1] = C_new[:,1]
        
        C_old = C_new
        
        end=time.time()
        
        RT=end-start
        if n%1==0 and n<50:
            saver(C_new,error)
        if n%10==0 and n>=50 and n<800:
            saver(C_new,error)
        if n%50==0 and n>=800 and n<1000:
            saver(C_new,error)
        if n%100==0 and n>=1000 and n<4000:
            saver(C_new,error)   
        elif n%300==0 and n>=4000:
            saver(C_new,error)
            
print("Total running time= {:.3f} sec".format(RT))