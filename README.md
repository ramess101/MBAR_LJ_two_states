# MBAR_LJ_two_states

MBAR_two_states.py is a simple code that reads in the energy files that are obtained from two different LJ parameter sets.
The code only uses the configurations from state 0 to predict the expectation values for state 1. These are then compared 
with the actual expectation values obtained from direct simulation of state 1.

The primary results are that the internal energies and liquid densities are poorly predicted in this case.
The explanation for this is that only a single configuration from state 0 has a non-negligible weight for state 1.
