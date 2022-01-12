
import numpy as np;
import math


# Virus particle deposition rate
def deposition(Parm, L, Q, U, r, phi, theta):
    '''
    pi    = 22/7;                   # pi value 
    d_p   = 1e-4;                 # particle diameter - cm
    rho_p = 1;                      # particle dencisy -  g / cm^3
    rho_f = 1.2e-3;                 # particle and flow dencisy -  g / cm^3
    g     = 980;                    # acceleration of gravity - cm / s^2
    mu    = 1.81e-4;                # airflow viscosity - g / cm . s
    lamda = 0.066e-4;               # mean free path of gas molecules - cm 
    k     = 1.38e-16;               # Boltzman constant
    T     = 23 + 273;               # Room Temperaturer  - K
    '''
    
    pi = Parm[0]; d_p = Parm[1]; rho_p = Parm[2]; rho_f = Parm[3]; g = Parm[4]; mu = Parm[5]; lamda = Parm[6]
    k_B = Parm[7]; T = Parm[8];
    
    # deposition due to diffusion 
    C_c = 1 - (2 * lamda / d_p) *  ( 1.257 + 0.4 * np.exp(-0.55 * (d_p / lamda))) # Cunningham slip correction factor 
    D   =  (C_c * k_B * T ) / (3 * pi * mu * d_p); # diffusion coefficient 
     
    B   = ( pi * D * L)/ ( 4 * Q );
    K_D = 1 - 0.819 * np.exp( (-1) * 14.63 * B) -0.0976 * np.exp( (-1) * 89.22 * B) - 0.0325 * np.exp( (-1) * 228* B) - 0.0509 * np.exp( (-1) * 125.9 * math.pow(B, 2/3) )
  
    # deposition due to Sedimentation         
    
    #w   = pi/2 - phi;  # inclination angle relative to gravity , phi - gravity angle       
    u_g = ( ( C_c * rho_p * g * math.pow(d_p,2) ) / (18 * mu) ) *  ( (rho_p - rho_f) / rho_p ); # settling velocity     
          
    e = (3 * pi * L * u_g) / (16* U * r)
    if e > 1:
        e = 1
    K_S = ( (2*Q)/(pi *L) ) *( 2* e * np.sqrt(1 - math.pow(e, 2/3)) - math.pow(e, 1/3) * np.sqrt(1 - math.pow(e, 2/3) ) +  math.asin( math.pow(e, 1/3) ) )
    
    
    # deposition due to impaction 
    
    #beta = (7.5/35 ) * theta  + 27.5;                     # effective branching angel, theta - actual branching angel
    d_L  = r;                                             # airway diameter
    St   = ( rho_p * math.pow( d_p, 2) * U * C_c) /(18 * mu * d_L); # stock number 
    

    K_I  = 1.3 * St/L
    
        
    K = K_D +  K_I + K_S ;
    K = [K_D, K_S, K_I]
    
    return K


# Airway partition
def Airway_Parti(L,n):
    L_cum = np.append(0,np.cumsum(L))
    X = np.zeros([len(L_cum)-1,n])   
    for i in range(1, len(L_cum)):
        rang = L_cum[i] - L_cum[i-1];
        x = np.arange(L_cum[ i -1],L_cum[ i ], rang/n );
        x = np.append(x[1:],L_cum[i]);
        X[i-1,:] = x

    return X

# Airflow velocity over airway generations 
def FlowVelocity(Parm, L, r):
    
    pi = Parm[0]; d_p = Parm[1]; rho_p = Parm[2]; rho_f = Parm[3]; g = Parm[4]; mu = Parm[5]; lamda = Parm[6]
    k_B = Parm[7]; T = Parm[8]; C_0 = Parm[9]; Q_0 = Parm[10]
    
    Z_o = 1e-5; Z_i = np.zeros(len(r));

    for i in range(1,len(r)+1):
        Z_l = Z_r = (8 * mu * L[len(r)-i])/(pi * math.pow(r[len(r)-i],4))
        Z_L = Z_l + Z_o; Z_R = Z_r + Z_o
        Z = Z_L * Z_R /( Z_L + Z_R )
        Z_o = Z
        Z_i[len(r)-i] = Z_o


    Q = np.zeros(len(r)); U = np.zeros(len(r));
        
    Q[0] = Q_0; U[0]= Q[0]/( pi * r[0]/2 *r[0]/2 );  
    
    for i in range(1,len(r)):
      
        Q_L = Q_0 * Z_i[i]/( Z_i[i] + Z_i[i] ); Q_R = Q_0 * Z_i[i]/( Z_i[i] + Z_i[i] );
        
        Q[i] = Q_L; U[i]= Q[i]/( pi * r[i]/2 * r[i]/2 ) 
        
        Q_0 = Q_R = Q_L;

    return [Q, U]

# Airway viral concentration
def Concen(Parm, U, K, P, x):
    '''
    pi   - pi value 
    d_p  - particle diameter - cm
    mu   - airflow viscosity - g / cm . s
    lamda - mean free path of gas molecules - cm 
    k_B   - Boltzman constant
    T     -  Absolute temperature 
    '''

    pi = Parm[0]; d_p = Parm[1]; mu = Parm[5]; lamda = Parm[6]
    k_B = Parm[7]; T = Parm[8]; C_0 = Parm[9]
    
    C_c = 1 - (2 * lamda / d_p) *  ( 1.257 + 0.4 * np.exp(-1.1 * (d_p / 2 * lamda))) # Cunningham slip correction factor 
    
    D   =  (C_c * k_B * T ) / (3 * pi * mu * d_p); # diffusion coefficient 
    
    
    B =  (U - np.sqrt( U**2 + 4 * D * (K+P)))/ ( 2 * D); 
    C = ( 2 * U * C_0 / (U + np.sqrt( U**2 + 4 * D * (K+P) )) )  * np.exp( B * x)
    
    return C

# Cell-limited Viral infection model
def model(z,t,b, d, p, c):
    U, I, V = z

    dUdt = (-1) * b * U * V;
    dIdt = b * U * V - d * I
    dVdt = p * I - c * V
    dzdt = [dUdt,dIdt, dVdt]
    return dzdt
# Immune response added viral infection model
def model1(z,t,b, d,p, K, c_T, c, s_T, r, m, k_T, d_T):
    U, I, V, T = z

    dUdt = (-1) * b * U * V;
    dIdt = b * U * V - d * I
    #dVdt = p*I  - c_T * V * T - c * V;
    dVdt = p*I + p * V * ( 1 - V/K) - c_T * V * T - c * V;
    dTdt = s_T + r * T * (V**m / (V**m + k_T**m) ) - d_T * T
    dzdt = [dUdt,dIdt, dVdt, dTdt ]
    return dzdt


