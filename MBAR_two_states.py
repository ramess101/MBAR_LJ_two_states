# -*- coding: utf-8 -*-
"""
Predicts the internal energy for the Model 1 and Model 0 potentials using configurations from just state 0
"""

import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR

P = 45. #[atm]
P *= 1.01325 #[bar]

M_w = 16.04 # [gm/mol]
N = 300.
N_A = 6.022e23 #[/mol]
nm3_to_ml = 10**21

bar_nm3_to_kJ_per_mole = 0.0602214086

R_g = 8.3144598 / 1000. #[kJ/mol/K]
T = 150. # [K]
beta = 1./(R_g*T)

# These are the LJ parameters used for state 0 and state 1
eps_0 = 1.5589 #[kJ/mol]
sig_0 = 0.38434 #[nm]
eps_1 = 1.5117 #[kJ/mol]
sig_1 = 0.38042 #[nm]

def rho_calc(V): # Calculate the density for a given volume
    return N * M_w * nm3_to_ml / N_A / V

def U_to_u(U,pV):
    u = beta*(U + pV)
    return u

# Import the potential energies and pressures for the different mdruns and reruns
# The notation is: subenergy"i_j" were "i" is the state that was used with "mdrun"
# and "j" is the state that was with used "rerun"

Up_00=np.loadtxt('subenergy0_0.txt')
U_00 = Up_00[:,1] #[kJ/mol]
pV_00 = Up_00[:,2] #[kJ/mol]
V_00 = pV_00 / P / bar_nm3_to_kJ_per_mole #[nm3]

Up_01 = np.loadtxt('subenergy0_1.txt')
U_01 = Up_01[:,1]
pV_01 = Up_01[:,2]
V_01 = pV_01 / P / bar_nm3_to_kJ_per_mole #[nm3]

Up_11 = np.loadtxt('subenergy1_1.txt')
U_11 = Up_11[:,1]
pV_11 = Up_11[:,2]
V_11 = pV_11 / P / bar_nm3_to_kJ_per_mole #[nm3]

# Convert potentials and pV to reduced internal energies
u_00 = U_to_u(U_00,pV_00)
u_01 = U_to_u(U_01,pV_01)
u_11 = U_to_u(U_11,pV_11)

# Using just the Model 0 mdrun samples
N_k = np.array([len(u_00),0]) # The number of samples from the two states
u_kn = np.array([u_00,u_01])
U_kn = U_00
V_kn = V_00
    
mbar = MBAR(u_kn,N_k)

(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)

# The first observable we are interested in is U, internal energy
A_kn = U_kn

(EA_k, dEA_k) = mbar.computeExpectations(A_kn)

# The second observable we are interested in is V, volume
A2_kn = V_kn

(EA2_k, dEA2_k) = mbar.computeExpectations(A2_kn)

# The third observable we are interested in is density
A3_kn = rho_calc(V_kn)
A3_kn = A3_kn[:]

(EA3_k, dEA3_k) = mbar.computeExpectations(A3_kn)

# The averages for the different mdruns and reruns

U_00_ave = np.mean(U_00)
U_11_ave = np.mean(U_11)
U_01_ave = np.mean(U_01)

V_00_ave = np.mean(V_00)
V_11_ave = np.mean(V_11)
V_01_ave = np.mean(V_01)

plt.plot(u_kn[0,:],label='Model 0')
plt.plot(u_kn[1,:],label='Model 1')
plt.ylabel('Reduced Potential Energy')
plt.xlabel('Sample')
plt.legend()
plt.show()

plt.plot(u_kn[1,:]-u_kn[0,:])
plt.ylabel(r'$\Delta$ Potential Energy')
plt.xlabel('Sample')
plt.legend()
plt.show()

plt.plot(U_kn[:N_k[0]+1],label='Model 0')
plt.plot(U_kn[N_k[0]:],label='Model 1')
plt.ylabel('Internal Energy (kJ/mol)')
plt.xlabel('Sample')
plt.legend()
plt.show()

plt.plot(V_kn[:N_k[0]+1],label='Model 0')
plt.plot(V_kn[N_k[0]:],label='Model 1')
plt.ylabel('Volume (nm$^3$)')
plt.xlabel('Sample')
plt.legend()
plt.show()

plt.plot(Deltaf_ij[0,:],label='Model 0')
plt.plot(Deltaf_ij[1,:],label='Model 1')
plt.ylabel('$\Delta$ F (kJ/mol)')
plt.xlabel('Model 0, Model 1')
plt.legend()
plt.show()

plt.errorbar(range(len(EA_k)),EA_k,dEA_k,label='MBAR')
plt.plot([U_00_ave,U_11_ave],label='Simulated')
plt.plot([U_00_ave,U_01_ave],label='Rerun')
plt.ylabel('Internal Energy (kJ/mol)')
plt.xlabel('Model')
plt.legend()
plt.show()

plt.plot(dEA_k)
plt.ylabel('Uncertainty Internal Energy (kJ/mol)')
plt.xlabel('Model')
plt.show()

plt.errorbar(range(len(EA2_k)),EA2_k,dEA2_k,label='MBAR')
plt.plot([V_00_ave,V_11_ave],label='Simulated')
plt.plot([V_00_ave,V_01_ave],label='Rerun')
plt.ylabel('Volume (nm$^3$)')
plt.xlabel('Model')
plt.legend()
plt.show()

plt.plot(dEA2_k)
plt.ylabel('Uncertainty Volume (nm$^3$)')
plt.xlabel('Model')
plt.show()

plt.errorbar(range(len(EA3_k)),EA3_k,dEA3_k,label='MBAR')
plt.plot([rho_calc(V_00_ave),rho_calc(V_11_ave)],label='Simulated')
plt.plot([rho_calc(V_00_ave),rho_calc(V_01_ave)],label='Rerun')
plt.ylabel(r'$\rho_l$ (gm/ml)')
plt.xlabel('Model')
plt.legend()
plt.show()

plt.plot(dEA3_k)
plt.ylabel(r'$\rho_l$ (gm/ml)')
plt.xlabel('Model')
plt.show()

plt.plot(mbar.W_nk[:,1])
plt.xlabel('Configuration')
plt.ylabel('Weight')
plt.title('Unsampled State')
plt.show()