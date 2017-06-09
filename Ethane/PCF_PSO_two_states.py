"""
PCF-PSO, predicts the internal energy for ethane given a PCF for a reference model
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import pandas as pd

# Physical constants
R_g = 8.314472e-3 # [kJ/mol/K]

#Simulation conditions

Temp = np.array([135.]) #[K]
rho_L = np.array([12.06117]) #[1/nm**3]
N = 400.

# Force field parameters from reference state

eps_ref = 98. #[K]
sig_ref = 0.375 #[nm]
lam_ref = 12.
      
# Gromacs output, pair correlation function for TraPPE

fname = 'TraPPE_PCF.txt'
xvg_ref = pd.read_fwf(fname, skiprows = 1, names=['r', 'RDF'])
RDF_ref = np.array(xvg_ref['RDF'])
r = np.array(xvg_ref['r'])

# Ensemble average
U_ens = np.array([-5522.483]) #[kJ/mol]

# Gromacs output, pair correlation function for Potoff

fname = 'Potoff_PCF.txt'
xvg_1= pd.read_fwf(fname, skiprows = 1, names=['r', 'RDF'])
RDF_1 = np.array(xvg_1['RDF'])
r_1 = np.array(xvg_1['r'])

# Gromacs output, pair correlation function for Mess-UP

fname = 'Mess_UP_PCF.txt'
xvg_2= pd.read_fwf(fname, skiprows = 1, names=['r', 'RDF'])
RDF_2 = np.array(xvg_2['RDF'])
r_2 = np.array(xvg_2['r'])

# Recall that Gromacs returns a single PCF for all CH3 interactions
    
N_PCFs = 1
N_sites = 2
N_pair = N_sites**2

# RDF bins

dr = r[1] - r[0]

# Simulation constants
r_c = r[-1] #[nm]

# Scaled constants

r_c_plus_ref = r_c / sig_ref

r_plus_ref = r/sig_ref

dr_plus_ref = dr/sig_ref


def RDF_smooth(RDF=RDF_ref):
    RDF_non_zero = RDF[RDF>0]
    RDF_zero = RDF[RDF==0]
    
    # Smooth the first two and last two points differently
    
    RDF_smoothed = RDF_zero
    
    RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF_non_zero[0] + 4.*RDF_non_zero[1] - 6.*RDF_non_zero[2] + 4.*RDF_non_zero[3] - RDF_non_zero[4]))
    RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF_non_zero[0] + 27.*RDF_non_zero[1] + 12.*RDF_non_zero[2] - 8.*RDF_non_zero[3] + 2*RDF_non_zero[4]))
    
    for j in range(2,len(RDF_non_zero)-2):
        RDF_smoothed = np.append(RDF_smoothed,1./35 * (-3.*RDF_non_zero[j-2] + 12.*RDF_non_zero[j-1] + 17.*RDF_non_zero[j] +12.*RDF_non_zero[j+1] - 3*RDF_non_zero[j+2]))

    RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF_non_zero[-1] + 27.*RDF_non_zero[-2] + 12.*RDF_non_zero[-3] - 8.*RDF_non_zero[-4] + 2*RDF_non_zero[-5]))
    RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF_non_zero[-1] + 4.*RDF_non_zero[-2] - 6.*RDF_non_zero[-3] + 4.*RDF_non_zero[-4] - RDF_non_zero[-5]))    
                
    RDF_smoothed[RDF_smoothed<0]=RDF[RDF_smoothed<0] # If initial point becomes negative set it to original value
    return RDF_smoothed

RDF_ref = RDF_smooth(RDF_ref)

def RDF_0(U,T):
    return np.exp(-U/T)

def RDF_hat_calc(RDF_real, RDF_0_ref,RDF_0_hat,print_RDFs=0):
    RDF_0_ref_zero = RDF_0_ref[RDF_0_ref<1e-2] # Using exactly 0 leads to some unrealistic ratios at very close distances
    RDF_0_ref_non_zero = RDF_0_ref[RDF_0_ref>1e-2]
    RDF_real_non_zero = RDF_real[RDF_0_ref>1e-2]
    RDF_ratio_non_zero = RDF_real_non_zero / RDF_0_ref_non_zero
    RDF_ones = np.ones(len(RDF_0_ref_zero))
    RDF_ratio = np.append(RDF_ones,RDF_ratio_non_zero)
    RDF_hat = RDF_ratio * RDF_0_hat
        
    if print_RDFs == 1:
        plt.scatter(r,RDF_real,label='Ref')
        plt.scatter(r,RDF_0_ref,label='Ref_0')
        plt.scatter(r,RDF_0_hat,label='Hat_0')
        plt.scatter(r,RDF_hat,label='Hat')
        plt.legend()
        plt.show()
    
    return RDF_hat

def U_Mie(r, e_over_k, sigma, n = 12., m = 6.):
    """
    The Mie potential calculated in [K]. 
    Note that r and sigma must be in the same units. 
    The exponents (n and m) are set to default values of 12 and 6    
    """
    C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
    U = C*e_over_k*((r/sigma)**-n - (r/sigma)**-m)
    U[U>1e10] = 1e10 # No need to store extremely large values of U that can lead to overflow
    return U

def r_min_calc_Mie(sig, n=12., m=6.):
    r_min = (n/m*sig**(n-m))**(1./(n-m))
    return r_min

def U_Corr_Mie(e_over_k, sigma, r_c, n = 12., m = 6.): #I need to correct this for m != 6
    C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
    #U = C*e_over_k*((1./(n-3))*r_c_plus**(3-n) - (1./3) * r_c_plus **(-3))*sigma**3 #[K * nm^3]
    # This is using the Gromacs tail corrections (assumes repulsive portion has gone to zero)
    C6 = C * e_over_k * sigma **6
    U = (-1./3) * C6 * r_c**(-3.)
    
    return U

def U_total_Mie(r, e_over_k, sigma, r_c, RDF, dr,  n = 12., m = 6.):
    U_int = (U_Mie(r, e_over_k, sigma,n,m)*RDF*r**2*dr).sum() # [K*nm^3]
    U_total = U_int + U_Corr_Mie(e_over_k,sigma,r_c,n,m)
    U_total *= 2*math.pi
    return U_total

def U_hat_Mie(eps_pred,sig_pred,lam_pred,method,RDF_ref,RDF_0_Temp_ref,r_ref=r,sig_ref=sig_ref,r_c=r_c,r_plus_ref = r_plus_ref, dr_plus_ref = dr_plus_ref, r_c_plus_ref = r_c_plus_ref,Temp=Temp):
    
    if method == 0: # Assumes constant r,RDF
        
        RDF = RDF_ref
        U_hat = U_total_Mie(r,eps_pred,sig_pred,r_c,RDF,dr,lam_pred) # Constant r_plus
        
    elif method == 1: # Assumes constant r* with respect to sigma,RDF
        
        RDF = RDF_ref
        U_hat = U_total_Mie(r_plus_ref*sig_pred,eps_pred,sig_pred,r_c_plus_ref*sig_pred,RDF,dr_plus_ref*sig_pred,lam_pred) # Constant r_plus
        
    elif method == 2: # Assumes constant r, predicts the zeroth order RDF
    
        RDF_0_Temp = RDF_0(U_Mie(r,eps_pred,sig_pred,lam_pred),Temp)
        RDF = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
        U_hat = U_total_Mie(r,eps_pred,sig_pred,r_c,RDF,dr,lam_pred) # Constant r_plus
    
    elif method == 3: # Assumes constant r* with respect to sigma, predicts the zeroth order RDF
    
        RDF_0_Temp = RDF_0(U_Mie(r_plus_ref*sig_pred,eps_pred,sig_pred,lam_pred),Temp)
        RDF = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
        U_hat = U_total_Mie(r_plus_ref*sig_pred,eps_pred,sig_pred,r_c_plus_ref*sig_pred,RDF,dr_plus_ref*sig_pred,lam_pred) # Constant r_plus
    
    elif method == 5: # Assumes constant r* with respect to rmin,RDF
        
        r_min_pred = r_min_calc_Mie(sig_pred,lam_pred)
        RDF = RDF_ref
        U_hat = U_total_Mie(r_plus_ref*r_min_pred,eps_pred,sig_pred,r_c_plus_ref*r_min_pred,RDF,dr_plus_ref*r_min_pred,lam_pred) # Constant r_plus
     
    elif method == 6: # Assumes constant r* with respect to rmin, predicts the zeroth order RDF
    
        r_min_pred = r_min_calc_Mie(sig_pred,lam_pred)
        RDF_0_Temp = RDF_0(U_Mie(r_plus_ref*r_min_pred,eps_pred,sig_pred,lam_pred),Temp)
        RDF = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
        U_hat = U_total_Mie(r_plus_ref*r_min_pred,eps_pred,sig_pred,r_c_plus_ref*r_min_pred,RDF,dr_plus_ref*r_min_pred,lam_pred) # Constant r_plus
    
    return U_hat

def U_hat_Mie_state(eps,sig,lam,method,RDF_ref=RDF_ref,rho_L=rho_L,Temp=Temp):
         
    RDF_0_Temp_ref = RDF_0(U_Mie(r,eps_ref,sig_ref,lam_ref),Temp)
        
    U_L = U_hat_Mie(eps,sig,lam,method,RDF_ref,RDF_0_Temp_ref)
    
    U_L *= R_g * rho_L * N
               
    U_L *= N_pair # Accounts for the four interactions (gromacs only supplies a single RDF)

    return U_L

U_error = lambda method: U_ens - U_hat_Mie_state(eps_ref,sig_ref,lam_ref,method) #Accounts for the difference between the ensemble average and the average obtained using PCF

def U_hat(eps,sig,lam,method):
        
    U_L_hat = U_hat_Mie_state(eps,sig,lam,method) + U_error(method)
    return U_L_hat
