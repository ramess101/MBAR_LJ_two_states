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

Temp = np.array([178, 197, 217, 236, 256, 275, 110, 135, 160, 290]) #[K]
rho_v = np.array([0.029258136, 0.073438124, 0.162222461, 0.308079559 ,0.563326801 ,0.974212647,0.00005,0.00122,0.00979,1.54641]) #[1/nm**3]
rho_L = np.array([11.05471434, 10.57313826, 10.02591012, 9.449821775, 8.750190843 ,7.926828949 ,12.61698,12.06117,11.48692,7.03602]) #[1/nm**3]

N = 400.

# Force field parameters from reference state

eps_ref = 98. #[K]
sig_ref = 0.375 #[nm]
lam_ref = 12.
      
# Gromacs output, pair correlation function for TraPPE

fname = 'TraPPE_all_PCFs.txt'
RDFs_ref = np.loadtxt(fname,delimiter='\t')
RDFs_ref = RDFs_ref[1:,:]
r = np.linspace(0.002,1.4,num=700) #[nm]

# Ensemble average
U_L_ens = np.array([-12.31010767,-11.63620643,-10.90281345,-10.15849685,-9.301811562,-8.355077371,-14.69796392,-13.80568346,-12.93521773,-7.393801775]) #[kJ/mol]
U_v_ens = np.array([-0.056372135,-0.123434941,-0.256256544,-0.460407723,-0.797075962,-1.304501055,-5.44979E-05,-0.003345935,-0.021188385,-1.970442484]) #[kJ/mol]
deltaU_ens = U_v_ens - U_L_ens

# Gromacs output for Potoff

eps_Potoff = 121.25 #[K]
sig_Potoff = 0.3783 #[nm]
lam_Potoff = 16.

U_L_ens_Potoff = np.array([-13.78517048,-12.9977746,-12.14174677,-11.27774358,-10.29840331,-9.215496054,-16.59808909,-15.54898814,-14.52675775,-8.14678519])
U_v_ens_Potoff = np.array([-0.069256206,-0.153756377,-0.309075253,-0.54333501,-0.92100389,-1.477751845,-0.000307792,-0.002866095,-0.027955288,-2.209428868])
deltaU_ens_Potoff = U_v_ens_Potoff - U_L_ens_Potoff

# Recall that Gromacs returns a single PCF for all CH3 interactions
    
N_PCFs = 1
N_sites = 2
N_pair = N_sites**2
N_columns = N_PCFs * 2

# RDF bins

dr = r[1] - r[0]

# Simulation constants
r_c = r[-1] #[nm]

# Scaled constants

r_c_plus_ref = r_c / sig_ref

r_plus_ref = r/sig_ref

dr_plus_ref = dr/sig_ref


def RDF_smooth(RDFs):
    RDFs_smoothed = np.empty(RDFs.shape)
    for i in range(RDFs.shape[1]):
        RDF = RDFs[:,i]
        RDF_non_zero = RDF[RDF>0]
        RDF_zero = RDF[RDF==0]
        if len(RDF_non_zero) > 5:
            # Smooth the first two and last two points differently
            
            RDF_smoothed = RDF_zero
            
            RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF_non_zero[0] + 4.*RDF_non_zero[1] - 6.*RDF_non_zero[2] + 4.*RDF_non_zero[3] - RDF_non_zero[4]))
            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF_non_zero[0] + 27.*RDF_non_zero[1] + 12.*RDF_non_zero[2] - 8.*RDF_non_zero[3] + 2*RDF_non_zero[4]))
            
            for j in range(2,len(RDF_non_zero)-2):
                RDF_smoothed = np.append(RDF_smoothed,1./35 * (-3.*RDF_non_zero[j-2] + 12.*RDF_non_zero[j-1] + 17.*RDF_non_zero[j] +12.*RDF_non_zero[j+1] - 3*RDF_non_zero[j+2]))
    
            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF_non_zero[-1] + 27.*RDF_non_zero[-2] + 12.*RDF_non_zero[-3] - 8.*RDF_non_zero[-4] + 2*RDF_non_zero[-5]))
            RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF_non_zero[-1] + 4.*RDF_non_zero[-2] - 6.*RDF_non_zero[-3] + 4.*RDF_non_zero[-4] - RDF_non_zero[-5]))    
                        
            RDF_smoothed[RDF_smoothed<0]=0
            RDFs_smoothed[:,i] = RDF_smoothed
        else:
            RDFs_smoothed[:,i] = RDF
        if Temp[int(i/2)] == min(Temp) and i%2 == 1: # Treat the vapor phase at lowest Temp differently
            RDF_smoothed = 1./70 * (69.*RDF[0] + 4.*RDF[1] - 6.*RDF[2] + 4.*RDF[3] - RDF[4])
            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF[0] + 27.*RDF[1] + 12.*RDF[2] - 8.*RDF[3] + 2*RDF[4]))
            
            for j in range(2,len(RDF)-2):
                RDF_smoothed = np.append(RDF_smoothed,1./35 * (-3.*RDF[j-2] + 12.*RDF[j-1] + 17.*RDF[j] +12.*RDF[j+1] - 3*RDF[j+2]))
    
            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF[-1] + 27.*RDF[-2] + 12.*RDF[-3] - 8.*RDF[-4] + 2*RDF[-5]))
            RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF[-1] + 4.*RDF[-2] - 6.*RDF[-3] + 4.*RDF[-4] - RDF[-5]))    
                        
            RDF_smoothed[RDF_smoothed<0]=RDF[RDF_smoothed<0]
            RDFs_smoothed[:,i] = RDF_smoothed
    return RDFs_smoothed

RDFs_ref = RDF_smooth(RDFs_ref)

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

def U_hat_Mie(eps_pred,sig_pred,lam_pred,method,RDF_ref,RDF_0_Temp_ref,t,r_ref=r,sig_ref=sig_ref,r_c=r_c,r_plus_ref = r_plus_ref, dr_plus_ref = dr_plus_ref, r_c_plus_ref = r_c_plus_ref,Temp=Temp):
    
    if method == 0: # Assumes constant r,RDF
        
        RDF = RDF_ref
        U_hat = U_total_Mie(r,eps_pred,sig_pred,r_c,RDF,dr,lam_pred) # Constant r_plus
        
    elif method == 1: # Assumes constant r* with respect to sigma,RDF
        
        RDF = RDF_ref
        U_hat = U_total_Mie(r_plus_ref*sig_pred,eps_pred,sig_pred,r_c_plus_ref*sig_pred,RDF,dr_plus_ref*sig_pred,lam_pred) # Constant r_plus
        
    elif method == 2: # Assumes constant r, predicts the zeroth order RDF
    
        RDF_0_Temp = RDF_0(U_Mie(r,eps_pred,sig_pred,lam_pred),Temp[t])
        RDF = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
        U_hat = U_total_Mie(r,eps_pred,sig_pred,r_c,RDF,dr,lam_pred) # Constant r_plus
    
    elif method == 3: # Assumes constant r* with respect to sigma, predicts the zeroth order RDF
    
        RDF_0_Temp = RDF_0(U_Mie(r_plus_ref*sig_pred,eps_pred,sig_pred,lam_pred),Temp[t])
        RDF = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
        U_hat = U_total_Mie(r_plus_ref*sig_pred,eps_pred,sig_pred,r_c_plus_ref*sig_pred,RDF,dr_plus_ref*sig_pred,lam_pred) # Constant r_plus
    
    elif method == 5: # Assumes constant r* with respect to rmin,RDF
        
        r_min_pred = r_min_calc_Mie(sig_pred,lam_pred)
        RDF = RDF_ref
        U_hat = U_total_Mie(r_plus_ref*r_min_pred,eps_pred,sig_pred,r_c_plus_ref*r_min_pred,RDF,dr_plus_ref*r_min_pred,lam_pred) # Constant r_plus
     
    elif method == 6: # Assumes constant r* with respect to rmin, predicts the zeroth order RDF
    
        r_min_pred = r_min_calc_Mie(sig_pred,lam_pred)
        RDF_0_Temp = RDF_0(U_Mie(r_plus_ref*r_min_pred,eps_pred,sig_pred,lam_pred),Temp[t])
        RDF = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
        U_hat = U_total_Mie(r_plus_ref*r_min_pred,eps_pred,sig_pred,r_c_plus_ref*r_min_pred,RDF,dr_plus_ref*r_min_pred,lam_pred) # Constant r_plus
    
    return U_hat

def U_hat_Mie_state(eps,sig,lam,method,RDF_all=RDFs_ref,rho_v=rho_v,rho_L=rho_L,Temp=Temp):
    
    U_L = np.empty(len(Temp))
    U_v = U_L.copy()
    
    for t in range(0, len(Temp)):
        rhov_Temp = rho_v[t]
        rhoL_Temp = rho_L[t]
        
        RDF_Temp_L = RDF_all[:,N_columns*t]
        RDF_Temp_v = RDF_all[:,N_columns*t+N_PCFs]
        
        RDF_0_Temp_ref = RDF_0(U_Mie(r,eps_ref,sig_ref,lam_ref),Temp[t])
        
        U_L_Temp = U_hat_Mie(eps,sig,lam,method,RDF_Temp_L,RDF_0_Temp_ref,t)
        U_v_Temp = U_hat_Mie(eps,sig,lam,method,RDF_Temp_v,RDF_0_Temp_ref,t)
        
        U_L_Temp *= R_g * rhoL_Temp
        U_v_Temp *= R_g * rhov_Temp
        
        U_L[t] = U_L_Temp
        U_v[t] = U_v_Temp
                                      
    U_L *= N_pair # Accounts for the four interactions (gromacs only supplies a single RDF)
    U_v *= N_pair # Accounts for the four interactions (gromacs only supplies a single RDF)

    return U_L, U_v

U_L_error = lambda method: U_L_ens - U_hat_Mie_state(eps_ref,sig_ref,lam_ref,method)[0] #Accounts for the difference between the ensemble average and the average obtained using PCF
U_v_error = lambda method: U_v_ens - U_hat_Mie_state(eps_ref,sig_ref,lam_ref,method)[1] #Accounts for the difference between the ensemble average and the average obtained using PCF
                                                    
def U_hat(eps,sig,lam,method):
        
    U_L_hat = U_hat_Mie_state(eps,sig,lam,method)[0] + U_L_error(method)
    U_v_hat = U_hat_Mie_state(eps,sig,lam,method)[1] + U_v_error(method)
    return U_L_hat, U_v_hat

U_L_PCF = np.zeros([len(Temp),8])
dev_U_L = U_L_PCF.copy()
U_v_PCF = U_L_PCF.copy()
dev_U_v = U_L_PCF.copy()
deltaU_PCF = U_L_PCF.copy()
dev_deltaU = U_L_PCF.copy()

methods=np.array(['Constant r, RDF','Constant r*, RDF','Constant r, Predicted RDF','Constant r*, Predicted RDF','Average of Constant r* Methods','Constant r* (with rmin), RDF', 'Constant r* (with rmin), Predicted RDF', 'Average of Constant r* (with rmin) methods'])

for i in range(0,8): # Loop through the different PCF-PSO methods
    if i == 4:
        U_L_PCF[:,i] = (U_L_PCF[:,1] + U_L_PCF[:,3])/2. # Method 1 is known to underpredict while method 3 is known to over predict, their average is the best estimate
        U_v_PCF[:,i] = (U_v_PCF[:,1] + U_v_PCF[:,3])/2. # Method 1 is known to underpredict while method 3 is known to over predict, their average is the best estimate
    elif i == 7:
        U_L_PCF[:,i] = (U_L_PCF[:,5] + U_L_PCF[:,6])/2. # Method 5 is known to underpredict while method 6 is known to over predict, their average is the best estimate
        U_v_PCF[:,i] = (U_v_PCF[:,5] + U_v_PCF[:,6])/2. # Method 5 is known to underpredict while method 6 is known to over predict, their average is the best estimate
    else:
        U_L_PCF[:,i] = U_hat(eps_Potoff,sig_Potoff,lam_Potoff,i)[0]
        U_v_PCF[:,i] = U_hat(eps_Potoff,sig_Potoff,lam_Potoff,i)[1]
    deltaU_PCF[:,i] = U_v_PCF[:,i] - U_L_PCF[:,i]
    dev_U_L[:,i] = (U_L_PCF[:,i] - U_L_ens_Potoff)/U_L_ens_Potoff*100.
    dev_U_v[:,i] = (U_v_PCF[:,i] - U_v_ens_Potoff)/U_v_ens_Potoff*100.
    dev_deltaU[:,i] = (deltaU_PCF[:,i] - deltaU_ens_Potoff)/ deltaU_ens_Potoff * 100.
    #dev_U_L[:,i] = (U_L_PCF[:,i] - U_L_ens_Potoff)/(U_L_ens_Potoff-U_L_ens)*100.
    #dev_U_v[:,i] = (U_v_PCF[:,i] - U_v_ens_Potoff)/(U_v_ens_Potoff-U_v_ens)*100.
    #dev_deltaU[:,i] = (deltaU_PCF[:,i] - deltaU_ens_Potoff)/ (deltaU_ens - deltaU_ens_Potoff) * 100.
    plt.scatter(Temp,dev_U_L[:,i],label=methods[i])
    #plt.scatter(Temp,U_L_PCF[:,i],label=methods[i])

#plt.scatter(Temp,U_L_ens,label='TraPPE')
#plt.scatter(Temp,U_L_ens_Potoff,label='Potoff')    
plt.xlabel('Temperature (K)')
#plt.ylabel('$U_l$ (kJ/mol)')
plt.ylabel('Percent Error in $U_l$')
plt.legend()
plt.show()

plt.scatter(Temp,dev_U_L[:,4],label='Hybrid using $\sigma$') 
plt.scatter(Temp,dev_U_L[:,7],label='Hybrid using $r_{min}$')
plt.xlabel('Temperature (K)')
plt.ylabel('Percent Error in $U_l$')
plt.legend()          
plt.show()

plt.scatter(Temp,U_L_ens,label='TraPPE')
plt.scatter(Temp,U_L_ens_Potoff,label='Potoff')
plt.scatter(Temp,U_L_PCF[:,4],label='Hybrid using $\sigma$') 
plt.scatter(Temp,U_L_PCF[:,7],label='Hybrid using $r_{min}$')
plt.xlabel('Temperature (K)')
plt.ylabel('$U_l$ (kJ/mol)')
plt.legend()          
plt.show()

plt.scatter(Temp,dev_deltaU[:,4],label='Hybrid using $\sigma$') 
plt.scatter(Temp,dev_deltaU[:,7],label='Hybrid using $r_{min}$')
plt.xlabel('Temperature (K)')
plt.ylabel('Percent Error in $\Delta U$')
plt.legend()          
plt.show()

plt.scatter(Temp,deltaU_ens,label='TraPPE')
plt.scatter(Temp,deltaU_ens_Potoff,label='Potoff')
plt.scatter(Temp,deltaU_PCF[:,4],label='Hybrid using $\sigma$') 
plt.scatter(Temp,deltaU_PCF[:,7],label='Hybrid using $r_{min}$')
plt.xlabel('Temperature (K)')
plt.ylabel('$\Delta U$ (kJ/mol)')
plt.legend()          
plt.show()