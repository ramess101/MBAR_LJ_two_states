"""
Predicts the internal energy for a new model (Potoff or Mess-UP) using simulation
just for the initial model (TraPPE). The TraPPE and Mess-UP models represent
an example of very good phase space overlap as they are very similar LJ models.
The Potoff model has very poor phase space overlap having a different value
of sigma and a repulsive exponent of 16. 

"""

import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR
import sys
import pdb
from PCF_PSO_two_states import *

# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
bar_nm3_to_kJ_per_mole = 0.0602214086
R_g = 8.3144598 / 1000. #[kJ/mol/K]

# Simulation compound, Ethane
M_w = 30.07 # [gm/mol]

# Constant temperature simulation
T = 135. # [K]
beta = 1./(R_g*T)

# Constant volume simulation
L = 3.21285 #[nm]
V = L**3 #[nm3]

# These are the Mie parameters used for state 0 (TraPPE) and state 1 (Potoff) and state 2 (Mess-UP)
eps_0 = 98. #[K]
sig_0 = 0.375 #[nm]
lam_0 = 12.
eps_1 = 121.29 #[K]
sig_1 = 0.3783 #[nm]
lam_1 = 16.
eps_2 = 98.5 #[K]
sig_2 = 0.375 #[nm]
lam_2 = 12.

def U_to_u(U,pV): #Converts internal energy, pressure, volume, and temperature into reduced potential energy
    u = beta*(U)
    return u

# Import the potential energies and pressures for the different mdruns and reruns
# The notation is: subenergy"i_j" were "i" is the state that was used with "mdrun"
# and "j" is the state that was with used "rerun"
# TraPPE is state 0, Potoff is state 1, Mess-UP is state 2

# Extract values for internal energy and pressure from gromacs output file for
# the reference state, i.e. the TraPPE model

#MRS: rewrite to include everything in the same matrix.
Up_00 = np.loadtxt('energy0_0.txt')
U_00 = Up_00[:,1] + Up_00[:,2] #[kJ/mol]
P_00 = Up_00[:,4] + Up_00[:,5]  #[bar]
pV_00 = V * P_00 * bar_nm3_to_kJ_per_mole #[kJ/mol]
Up_01 = np.loadtxt('energy0_1.txt')
Up_11 = np.loadtxt('energy1_1.txt')
U_01 = Up_01[:,1] + Up_01[:,2]
P_01 = Up_01[:,4] + Up_01[:,5]
pV_01 = V * P_01 * bar_nm3_to_kJ_per_mole #[kJ/mol]
U_11 = Up_11[:,1] + Up_11[:,2]
P_11 = Up_11[:,4] + Up_11[:,5]
pV_11 = V * P_11 * bar_nm3_to_kJ_per_mole #[kJ/mol]
Up_02 = np.loadtxt('energy0_2.txt')
Up_22 = np.loadtxt('energy2_2.txt')
U_02 = Up_02[:,1] + Up_02[:,2]
P_02 = Up_02[:,4] + Up_02[:,5]
pV_02 = V * P_02 * bar_nm3_to_kJ_per_mole #[kJ/mol]
U_22 = Up_22[:,1] + Up_22[:,2]
P_22 = Up_22[:,4] + Up_22[:,5]
pV_22 = V * P_22 * bar_nm3_to_kJ_per_mole #[kJ/mol]

# Assign variables to specific values from gromacs output

# Convert potentials and pV to reduced internal energies
u_00 = U_to_u(U_00,pV_00)
u_01 = U_to_u(U_01,pV_01)
u_11 = U_to_u(U_11,pV_11)
u_02 = U_to_u(U_02,pV_02)
u_22 = U_to_u(U_11,pV_22)

# Using just the Model 0 mdrun samples
N_k = np.array([len(u_00),0,0]) # The number of samples from the two states

u_kn = np.array([u_00,u_01,u_02])
U_kn = U_00

mbar = MBAR(u_kn,N_k)

(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
print "effective sample numbers"

print mbar.computeEffectiveSampleNumber()
#MRS: 1001 for state 0, 1.00 for state 2, 985 for state 3

# MRS: The observable we are interested in is U, internal energy.  The
# question is, WHICH internal energy.  We are interested in the
# internal energy generated from the ith potential.  So there are
# actually _three_ observables.

# Now, the confusing thing, we can estimate the expectation of the
# three observables in three different states. We can estimate the
# observable of U_0 in states 0, 1 and 2, the observable of U_1 in
# states 0, 1, and 2, etc.


EAk = np.zeros([3,3])
dEAk = np.zeros([3,3])
(EAk[:,0], dEAk[:,0]) = mbar.computeExpectations(U_00) # potential energy of 0, estimated in state 0:2 (sampled from just 0)
(EAk[:,1], dEAk[:,1]) = mbar.computeExpectations(U_01) # potential energy of 1, estimated in state 0:2 (sampled from just 0) 
(EAk[:,2], dEAk[:,2]) = mbar.computeExpectations(U_02) # potential energy of 2, estimated in state 0:2 (sampled from just 0) 

# MRS: Some of these are of no practical importance.  We are most
# interested in the observable of U_0 in the 0th state, U_1 in the 1st
# state, and U_2 in the 2nd state, or the diagonal of the matrix EA.
EA = EAk.diagonal()
dEA = dEAk.diagonal()

#RM: Perhaps I am implementing MBAR improperly. Here I have reevaluated the expectation values
# by multiplying the weight of each configuration by the expectation value of the
# given configuration using state "i" rather than state 0.

EA_alt = np.zeros(3)

EA_alt[0] = np.mean(mbar.W_nk[:,0]*U_00)*len(U_00) #Weighted average
EA_alt[1] = np.mean(mbar.W_nk[:,1]*U_01)*len(U_01)
EA_alt[2] = np.mean(mbar.W_nk[:,2]*U_02)*len(U_02)

#MRS: Also easier to think of it just as this (same numbers)
EA_alt[0] = np.sum(mbar.W_nk[:,0]*U_00)
EA_alt[1] = np.sum(mbar.W_nk[:,1]*U_01)
EA_alt[2] = np.sum(mbar.W_nk[:,2]*U_02)

#MRS: Now the diagonal of EA matches up with the EA_alt! 

# Now look at the direct averages, and the "all weights=1" estimators

U_direct = np.zeros(3)
U_W1 = np.zeros(3)
U_PCF = np.zeros([3,8])
dU_direct = np.zeros(3)
dU_W1 = np.zeros(3)

U_direct[0] = np.mean(U_00)
U_direct[1] = np.mean(U_11)
U_direct[2] = np.mean(U_22)

dU_direct[0] = np.std(U_00)/np.sqrt(len(U_00)-1)
dU_direct[1] = np.std(U_11)/np.sqrt(len(U_11)-1)
dU_direct[2] = np.std(U_22)/np.sqrt(len(U_22)-1)

U_W1[0] = np.mean(U_00)
U_W1[1] = np.mean(U_01)
U_W1[2] = np.mean(U_02)

dU_W1[0] = np.std(U_00)/np.sqrt(len(U_00)-1)
dU_W1[1] = np.std(U_01)/np.sqrt(len(U_01)-1)
dU_W1[2] = np.std(U_02)/np.sqrt(len(U_02)-1)

for i in range(0,8): # Loop through the different PCF-PSO methods
    if i == 4:
        U_PCF[0,i] = (U_PCF[0,1] + U_PCF[0,3])/2. # Method 1 is known to underpredict while method 3 is known to over predict, their average is the best estimate
        U_PCF[1,i] = (U_PCF[1,1] + U_PCF[1,3])/2.
        U_PCF[2,i] = (U_PCF[2,1] + U_PCF[2,3])/2.
    elif i == 7:
        U_PCF[0,i] = (U_PCF[0,5] + U_PCF[0,6])/2. # Method 5 is known to underpredict while method 6 is known to over predict, their average is the best estimate
        U_PCF[1,i] = (U_PCF[1,5] + U_PCF[1,6])/2.
        U_PCF[2,i] = (U_PCF[2,5] + U_PCF[2,6])/2.
    else:
        U_PCF[0,i] = U_hat(eps_0,sig_0,lam_0,i)
        U_PCF[1,i] = U_hat(eps_1,sig_1,lam_1,i)
        U_PCF[2,i] = U_hat(eps_2,sig_2,lam_2,i)

for i in range(0,3):
    print "For state {:d}:".format(i) 
    print "    MBAR estimate for U =  {:10.3f} +/- {:10.3f}".format(EA[i],dEA[i])
    print "    Direct average      =  {:10.3f} +/- {:10.3f}".format(U_direct[i],dU_direct[i])
    print "    Weights=1 estimate  =  {:10.3f} +/- {:10.3f}".format(U_W1[i],dU_W1[i])
    print "    PCF-PSO, general  =  {:10.3f} ".format(U_PCF[i,0])
    print "    PCF-PSO, scaled  =  {:10.3f} ".format(U_PCF[i,1])
    print "    PCF-PSO, predicted zeroth order  =  {:10.3f} ".format(U_PCF[i,2])
    print "    PCF-PSO, scaled, predicted zeroth order  =  {:10.3f} ".format(U_PCF[i,3])
    print "    PCF-PSO, hybrid of scaled methods  =  {:10.3f} ".format(U_PCF[i,4])
    print "    PCF-PSO, scaled by rmin =  {:10.3f} ".format(U_PCF[i,5])
    print "    PCF-PSO, scaled by rmin, predicted zeroth order  =  {:10.3f} ".format(U_PCF[i,6])
    print "    PCF-PSO, hybrid of scaled by rmin methods  =  {:10.3f} ".format(U_PCF[i,7])

# MRS: amusing: find the "amount of weight" that guarantees at least 10%
# effective samples.  For 2, it's obviously lambda=1 (use MBAR
# weights). What is it for state 1, and what are the averages in that
# state?

# find the lambda that gets the desired number of effective samples

def balanceweights(lam):
    W_nk = np.exp(lam*mbar.Log_W_nk[:,1])
    W_nk /= np.sum(W_nk)
    N_eff = 1.0/((W_nk**2).sum())
    return N_eff - N_eff_target

# set percent to 5%
N_eff_target = N_k[0]*0.05

import scipy.optimize
rightlambda = scipy.optimize.brentq(balanceweights,0,1)

W_nk = np.exp(rightlambda*mbar.Log_W_nk[:,1])
W_nk /= np.sum(W_nk)
N_eff = 1.0/((W_nk**2).sum())

EA_scaled = np.sum(W_nk*U_01)
dEA_scaled = np.std(U_01)/np.sqrt(N_eff-1)

# MRS: there's a formula for uncertainty with weighted averages, but I
# can't recall it on the plane. I THINK it going to be about the
# standard estimator, scaled by # of effective samples.  TODO: look up correct formula.

print "Estimator scaled for N_eff = 5% of total samples: {:10.3f}+/-{:10.3f}".format(EA_scaled, dEA_scaled)

#MRS: Now do bootstrapping for errors.
Nboots = 2000  # you can get pretty close with N=200
EAb = np.zeros([3,Nboots])
dEAb = np.zeros([3,Nboots])
rand_select = np.random.randint(0,N_k[0],size=[N_k[0],Nboots])

for n in range(Nboots):
    u_kn = np.array([u_00[rand_select[:,n]],u_01[rand_select[:,n]],u_02[rand_select[:,n]]])
    mbar = MBAR(u_kn,N_k)
    (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
    for i in range(3):
        tmpa, tmpb = mbar.computeExpectations(u_kn[0,:]) # potential energy of i, estimated in state 0:2 (sampled from just 0)
        (EAb[i,n], dEAb[i,n]) = (tmpa[i]/beta,tmpb[i]/beta) 

print "Value /   +/-   Bootstrapped uncertainty vs analytical uncertainty"
for i in range(3):
    print "    MBAR estimate for U in state {:d} =  {:10.3f} +/- {:10.3f} vs {:10.3f}".format(i,EA[i],np.std(EAb[i,:]),dEA[i])
# pretty good analytical estimates for 0 and 2 (not surprising since N_eff is near N!  Sample run I get:
#    MBAR estimate for U =   -5522.483 +/-      0.868 vs      0.873
#    MBAR estimate for U =   -5876.616 +/-      4.604 vs      0.002
#    MBAR estimate for U =   -5554.139 +/-      0.881 vs      0.889

# Plot the predicted RDF for Potoff

r_min_1 = r_min_calc_Mie(sig_1,lam_1)
r_min_0 = r_min_calc_Mie(sig_0,lam_0)

RDF_0_Temp_ref = RDF_0(U_Mie(r,eps_0,sig_0,lam_0),Temp)
RDF_0_Temp = RDF_0(U_Mie(r,eps_1,sig_1,lam_1),Temp)
RDF_predicted_zeroth = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp)
RDF_0_Temp_scaled = RDF_0(U_Mie(r_plus_ref*sig_1,eps_1,sig_1,lam_1),Temp)
RDF_scaled_predicted_zeroth = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp_scaled)
RDF_0_Temp_scaled_rmin = RDF_0(U_Mie(r_plus_ref*sig_0/r_min_0*r_min_1,eps_1,sig_1,lam_1),Temp)
RDF_scaled_predicted_zeroth_rmin = RDF_hat_calc(RDF_ref,RDF_0_Temp_ref,RDF_0_Temp_scaled_rmin)

plt.plot(r,RDF_predicted_zeroth,label='Predicted Zeroth Potoff')
plt.plot(r_plus_ref*sig_1,RDF_scaled_predicted_zeroth,label='Scaled Predicted Zeroth Potoff')
plt.plot(r_plus_ref*sig_0/r_min_0*r_min_1,RDF_scaled_predicted_zeroth_rmin,label='Scaled by rmin Predicted Zeroth Potoff')
plt.plot(r,RDF_ref,label='Reference (TraPPE)')
plt.plot(r_1,RDF_1,label='Actual Potoff')
plt.legend()
plt.xlabel('r (nm)')
plt.ylabel('PCF')
plt.show()

plt.plot(r,RDF_predicted_zeroth,label='Predicted Zeroth Potoff')
plt.plot(r_plus_ref*sig_1,RDF_scaled_predicted_zeroth,label='Scaled Predicted Zeroth Potoff')
plt.plot(r_plus_ref*sig_0/r_min_0*r_min_1,RDF_scaled_predicted_zeroth_rmin,label='Scaled by rmin Predicted Zeroth Potoff')
plt.plot(r,RDF_ref,label='Reference (TraPPE)')
plt.plot(r_1,RDF_1,label='Actual Potoff')
plt.legend()
plt.xlabel('r (nm)')
plt.ylabel('PCF')
plt.xlim([0.3,0.45])
plt.show()

plt.plot(r,RDF_predicted_zeroth,label='Predicted Zeroth Potoff')
plt.plot(r_plus_ref*sig_1,RDF_scaled_predicted_zeroth,label='Scaled Predicted Zeroth Potoff')
plt.plot(r_plus_ref*sig_0/r_min_0*r_min_1,RDF_scaled_predicted_zeroth_rmin,label='Scaled by rmin Predicted Zeroth Potoff')
plt.plot(r,RDF_ref,label='Reference (TraPPE)')
plt.plot(r_1,RDF_1,label='Actual Potoff')
plt.legend()
plt.xlabel('r (nm)')
plt.ylabel('PCF')
plt.xlim([0.4,1.4])
plt.ylim([0.5,1.5])
plt.show()

