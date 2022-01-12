#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 01:43:51 2022

@author: dixon
"""


import numpy as np;
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Deposition import deposition, Concen, Airway_Parti,FlowVelocity, model, model1

r = np.array([1.8, 1.22, .83, .56, .45, .35, .28, .23, .186, .154, .130, .109, .095, .082, .074, .061, .06, .054, .05, .047, .045, .043, .041, .04])
L = np.array([12, 4.76, 1.9, .76, 1.27, 1.07, .9, .76, .64, .54, .46, .39, .33, .27, .23, .2, .165, .14, .12, .099, .083, .07, .059, .05])
phi = np.array([0, 20, 31, 43, 39, 39, 40, 36, 39, 45, 43, 45, 45, 60, 60, 60, 60, 1, 1, 1, 1, 1, 1, 1 ]) # gravity angle
theta = np.array([0, 33, 34, 22, 20, 18, 19, 22, 28, 22, 33, 34, 37, 39, 39, 51, 45,1, 1, 1, 1, 1, 1, 1 ]); # effective branching angle

Z_o = 1e-5; Z_i = np.zeros(len(r));


pi    = 22/7;                     # pi value 
d_p   = 70e-4;                    # particle diameter - cm
rho_p = 1.18e-2;                  # particle dencisy -  g / cm^3
rho_f = 1.2e-3;                   # particle and flow dencisy -  g / cm^3
g     = 980 * ( (3600 * 24)**2 ); # acceleration of gravity - cm / day^2
mu    = 1.81e-4 * (24 * 3600);    # airflow viscosity - g / cm . day
lamda = 0.066e-4;                 # mean free path of gas molecules - cm 
k     = 1.38e-16*((3600* 24)**2); # Boltzman constant - cm^2 g / s^2 K
T     = 25 + 273;                 # Room Temperaturer  - K
C_0   = C_01 = 10**5.01           # Initial concentration - copies/cm^3
Q_0   = ( 30 * 1e+3 )*( 60 * 24); # cm^3/day
P     = 8.2/1;                    # Copies/cm^3 day−1 cell−1

Parm = np.array([pi, d_p, rho_p, rho_f, g, mu, lamda, k, T, C_0, Q_0])
    
Q, U = FlowVelocity(Parm, L, r)

dep  =  np.zeros([3,len(r)])
for i in range(0,len(r)):
    dep[:,i] = deposition( Parm, L[i], Q[i], U[i], r[i], phi[i], theta[i])   

#  K - particle flus lost to the airway surface per unit iength of the airway per unit time. 
K = [sum(x) for x in zip(*dep)]*L;  # / s^{-1}



################## ACE2 and virus concenrtrations #################################

# cumulative airway length
a = np.array([0]); L_new1 = np.cumsum(L); L_new1 = np.append(a,L_new1) 

# Concentration in each airway generation
C_new = [C_01]; t2 = np.linspace(0,35)
   
Parm[9] = C_01
    
for i in range(len(L)):
    c_new = Concen(Parm, U[i], K[i], P, L[i]) 

    C_new = np.append( C_new,Parm[9] - c_new ); 
    Parm[9] = c_new/2



#======================= ACE concentration in each airway  ===========================
ACE1 = np.random.normal(6.83,0.91,len(C_new[:12])); ACE1 = sorted(ACE1, reverse= True);
ACE2 = np.random.normal(5.83,0.71,len(C_new[12:])); ACE2 = sorted(ACE2, reverse= False);
ACE = np.append(ACE1, ACE2)

ACE_1 = ACE

################## Original patient data with differnt immune response #################################

t2 = np.linspace(0,21)
b = 3.97*1e-7; d= 4.71; p= 8.2; c= 0.6; ACE  = 10**7; I = 0.0; V = 5.01

z0 = z01 = [ ACE, I, V ]   
z = odeint( model, z0, t2, args= (b, d, p, c) )


def model(z,t,b, d,p, K, c_T, c, s_T, r, m, k_T, d_T):
    U, I, V, T = z

    dUdt = (-1) * b * U * V;
    dIdt = b * U * V - d * I
    dVdt = p * V * (1 - V/K)- c_T * V * T - c * V;
    dTdt = s_T + r * T * (V**m / (V**m + k_T**m) ) - d_T * T
    dzdt = [dUdt, dIdt, dVdt, dTdt ]
    return dzdt

b = 3.97*1e-7;d= 4.71; 
p = 8.2; K = 10**8; c_T = 5.01e-8; c = .6; r = 5.89; m = 2; k_T = 7.94e+7; d_T = 2.9e-2; T_0 = 1097.5
s_T = T_0 * d_T;

ACE  = 10**7; I = 0.0; V = 5.01

z_0 = [ ACE, I, V, T_0]
t2 = np.linspace(0,21)

L = np.array([1,3,5,7])

z_c = odeint( model1, z_0, t2, args= (b,d, p, K, c_T, c, s_T, r, m, k_T, d_T) )

z_c = np.abs(z_c)


# Virus load over airway generations 
fig = plt.figure(figsize = (8, 6))

plt.bar(L_new1, np.log10( C_new ), width = 1, alpha = 0.5); 
plt.xlabel('Airway Depth (cm)', fontsize = 18); 
plt.ylabel('Virus Concentration(C) $log_{10} $ (Copies/ml)', fontsize = 18)
plt.xticks(fontsize= 14); plt.yticks(fontsize=14);
plt.grid(); 
plt.tight_layout() 


# ACE2 distribution over airway generations 
fig = plt.figure(figsize = (8, 6))
plt.bar(np.arange(0,25),  ACE_1, width = .5, alpha = 0.5); 
plt.xlabel('Airway Generation Number(G)', fontsize = 18); 
plt.ylabel('ACE2 ($A$) ($log_{2} $ (Copies/ml)', fontsize = 18)
plt.xticks(fontsize= 14); plt.yticks(fontsize=14);
plt.grid(); 
plt.tight_layout()


# Viral infection dynamics over time
fig = plt.figure(figsize = (8, 6))
ax = plt.subplot(1,1,1)
ax.plot(t2,np.log10(z[:,0]),'b-',label='Susceptible (A)')
ax.plot(t2,np.log10(z[:,1]),'r--',label='Infected (I)')
ax.plot(t2,np.log10(z[:,2]),'k-*',label='Virus (C)')
ax.legend(fontsize = 12, frameon = False, ncol = 1)
ax.grid();
#plt.title('log')
ax.set_ylabel('Total Concentration $log_{10}$ (Copies/ml)', fontsize = 18)
ax.set_xlabel('Time (day)', fontsize = 18)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14);
plt.xticks(np.arange(0,22, 3.5),fontsize = 14)


ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot( t2, np.log10(z_c[:,3]), '.--', color="green", label = 'Immune (T)')
plt.legend(fontsize = 15, frameon = False, ncol = 1, loc = 4)
plt.yticks(fontsize = 14);
ax2.set_ylabel(' Total T-Cell Concentration ($T$) $log_{10}$ ($Copies/\mu l$)', fontsize = 18, color="green")
plt.tight_layout()
