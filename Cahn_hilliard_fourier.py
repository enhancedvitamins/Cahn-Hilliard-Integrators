from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.fft import rfft2, irfft2, fftfreq, rfftfreq
import matplotlib.animation as animation


a = 0.01 #gradient energy coefficient                                                                                
dt = 1e-5 #time step

"Selection of the type os scheme and the number of nodes to solve"
p=int(input("Select the Scheme: Euler(0) or BDF/AB(1): "))
if p==0:
    method=str("Euler-Spectral Scheme")
    if dt<=1.8e-7:
        t =0.0125
        m=1
    else:
        t=1.5e-5
        m=1
else:
    method=str("2nd order BDF/AB Scheme")
    t = 0.016
    m=int(input("Select the number of nodes: 128(1) or 254(2) 384(3): "))
    while (True):
        if m==1 or m==2 or m==3:
            break
        else:
            m=int(input("Select the number of nodes: 128(1) or 254(2) 384(3):"))

"the number of nodes depend of the user selection DO NOT CHANGE!!!"
nodes = int(128*m)
N_it = int(t/dt) #Number of iterations

"Fourier transform methods"
x, dx = np.linspace(0, 1, nodes, retstep=True) #initializating the sampling vector in the space domain
k1=rfftfreq(nodes, dx/(2*np.pi)) #sampling vector in fourier space for horizontal axis, to avoid double calculation we use the half dimension since we use rfft
k2=fftfreq(nodes, dx/(2*np.pi)) #sampling vector in fourier space for vertical axis, we use the wholde dimension using fft
K1,K2=np.meshgrid(k1,k2) #finding the 2D matrix of the horizontal and vertical wavelengths 
K=K1**2+K2**2 #square the waveleghnt K

"Initial random value of the fiel in the fourier space"
C_old=(np.random.uniform(-1,1,(nodes,nodes)))
C_old=rfft2(C_old)

"laplacian function"
def laplacian(C):
    C_h=(-K*(C))
    return C_h
"euler spectral function"
def euler(C):
    Cn=C + dt*laplacian(rfft2(irfft2(C)**3)-C-(a**2)*laplacian(C))
    return Cn
"ﬁrst-order semi-implicit Fourier spectral scheme "
def dbf(C):
    f=rfft2(irfft2(C)**3)-C
    A=(1+dt*(a*K)**2)
    B=((C)-dt*K*(f))
    Cn_h=B/A
    return Cn_h
"high-order semi-implicit Fourier spectral scheme "
def dbf_ad(Cc,Co):
    fo=rfft2(irfft2(Co)**3)-Co
    fc=rfft2(irfft2(Cc)**3)-Cc
    A=(3+2*dt*(a*K)**2)
    B=4*(Cc)-(Co)-2*dt*K*(2*(fc)-(fo))
    Cn_h=B/A
    return Cn_h

"ploting and animation function"
def saver(C_n,C_o):
    "Error definition"
    error = abs(irfft2(C_n - C_o))/nodes
    errormax = np.max(error)
    "ploting evolution"
    im1 = ax1.imshow(irfft2(C_n), vmin = -1, vmax = 1, cmap="brg") 
    ax1.invert_yaxis()
    cb1 = fig.colorbar(im1, ax=ax1,orientation="horizontal", anchor=(0, 0.01), shrink=0.6, pad=0.05)
    cb1.set_label('Species concentration')
    ax1.set_title("Concentration Evolution",loc='center',fontsize=(15))
    title1=ax1.text(40*m,140*m, "",fontsize=(20),weight="bold")
    title1.set_text("Cahn-Hilliard {} ($\Delta$t={:.1e} s)".format(method,dt))
    title2 = ax1.text(.5, .1, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax1.transAxes, ha="center",weight="bold")
    title2.set_text("Time= {:.2e}".format(t_ac))
    "ploting error"
    im2 = ax2.imshow(error*100,vmin = 0, vmax = "{:5e}".format(errormax*100), cmap="brg") 
    ax2.invert_yaxis()
    cb2 = fig.colorbar(im2, ax=ax2, orientation="horizontal",anchor=(0, 0.01), shrink=0.6, pad=0.05)
    cb2.set_label('Error %')
    ax2.set_title("Error Evolution",loc='center',fontsize=(15))
    title3 = ax2.text(.5, .1, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax2.transAxes, ha="center",weight="bold")
    title3.set_text("Max Error= {:.2e}%".format(errormax*100))
    title4=ax1.text(120*m, -20*m, "",bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},color="b",fontsize=(13),weight="bold")
    title4.set_text("CPU_time={:.2f} s".format(RT))
    plt.show()
    file.grab_frame()
    print("time=",t_ac) 
    cb1.remove()
    cb2.remove()
    ax1.clear()
    ax2.clear() 

"initialise of the figure"
fig ,(ax1,ax2)= plt.subplots(1,2,figsize=(14,8))        
"save the file"
file = animation.FFMpegWriter(fps=5)
with file.saving(fig, "Cahn-hilliard BDF-AB nodes=384 dt=1e-5.mp4", 200):
    start=time.time()
    for n in range (0,N_it):
        t_ac= n*dt
        if p==1: 
            if n==0:
                C_new=dbf(C_old)
                C_cur=C_new
            elif n>0:
                C_new=dbf_ad(C_cur,C_old)
                C_old=C_cur
                C_cur=C_new
        elif p==0:
            if n==0:
                C_new=euler(C_old)
                C_cur=C_new
            elif n>0:
                C_new=euler(C_cur)
                C_old=C_cur
                C_cur=C_new
        end=time.time()
        RT=end-start
           
        if n%1==0 and n<20:
            saver(C_new,C_old)
        if n%10==0 and n>=20 and n<200:
            saver(C_new,C_old)
        if n%50==0 and n>=200 and n<1000:
            saver(C_new,C_old)
        if n%100==0 and n>=1000 and n<4000:
            saver(C_new,C_old)   
        elif n%200==0 and n>=4000:
            saver(C_new,C_old)
            
print("Total running time= {:.3f} sec".format(RT))  
plt.show()