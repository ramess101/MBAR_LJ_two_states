#!/usr/bin/python -wd

import numpy as np
import subprocess
import pymbar
import pdb
import os
import sys
import mdtraj as md
import shutil

plot = True
nenergies = 25000  # should be read from the files, not hard coded
nequil = 1000 # don't use the first 1000 energies structures. Will need to be adjusted / controlled.
nevery = 10 # the structures are only every 10 energy file steps.
directory = "select/jobs/";  # will need to be adjusted
# volume mapping is for pressure differences, harmonic for temperature differences.
# methods applied are by selecting an array that has both.
mapping = ['volume','harmonic']  # both harmonic and volume
#mapping = ['volume'] # just volume
#mapping = ['harmonic'] # just harmonic
#mapping = [] # no mapping
integration = 'direct'
type = 'FCC' 
dirtext = 'LJ_' + type + '_512_0REPLACETK_REPLACEPP_NPT__100G_0L';
npermol = 216  # number of molecules in the system (used to calculate DOF).
dirbase = os.path.join(directory,dirtext)
# Tarray is the array of temperatures, Parray the array of temperatures.  The 2D grid will be covered.
Tarray = [30,40]
Parray = [20400, 24800]
KB = 0.008314472 #kJ/mol*K  # important to get these right. . . 
Pconv = 0.060221415  # conversion factor from bar*nm^3 to kJ/mol.
          # 1 bar * m^3 = 10^5 Pa * m^3 = 10^5 N/m^2 * m^3 = 10^5 N*m = 10^5 J = 10^2 kJ
          # so 1 bar*m^3 = 10^2 kJ, so kJ/bar m^3 = 10^2. so (kJ/bar m^3) * 1 m^3 / 10^27 nm^3 =
          # kJ/(bar*nm^3) = 10^-25 kJ/(bar*nm^3) x 6.02215 x 10^23 things/mol -> (kJ/mol)/*(bar nm^3) = 6.02215 x 10-2)

dirnames = list()
Tpoints = np.zeros([len(Tarray)*len(Parray)])
Ppoints = np.zeros([len(Tarray)*len(Parray)])

k = 0
for T in Tarray:
    for P in Parray:
        dirnames.append((dirbase.replace('REPLACET',str(T))).replace('REPLACEP',str(P)))
        Tpoints[k] = T
        Ppoints[k] = P
        k+=1

nstates = k
#gromacsbin = '/Users/mrshirts/work/gromacs_allv/git/gromacs_averaging/install/bin/gmx'
#gromacsbin = '/Users/mrshirts/work/gromacs_allv/gromacs-5.0/install/bin/gmx_d'
gromacsbin = '/Users/mrshirts/work/gromacs_allv/gromacs-5.1.3/install/bin/gmx_d'
writevolumes = 'volumes.txt' 
threadinfo = ' -nt 1 '
Lsize = np.zeros(nstates,float)  # the scaling factor, used for mapping.
# we need to go into each directory:
   # 1. calculate the average volume
   # 2. read the gro
# for each FROM (directory)
#  for each TO (includes unsampled states - linearly interpolate the energy
#      calculate the transformation
#      write the new gro
#      calculate the jacobian
#      do a rerun
#      construct the u_kln
#      run MBAR 
#  process MBAR             

npoints = (nenergies-nequil)/nevery+1

u_kln = np.zeros([nstates,nstates,npoints],float)

#Writes out the volumes so we can see what they are for scaling.
f = open(writevolumes,'w')
for i,d in enumerate(dirnames):
    # get the average volume for this simulation

    volume_file = open(os.path.join(d,'equil_volume.xvg'),'r')
    lines = volume_file.readlines()
    volume_file.close()
    volumes = np.zeros(len(lines)-22) # hard coded header - may need to be changed.
    j = 0
    for l in lines:
        if l[0] != '#' and l[0] != '@':
            volumes[j] = float(l.split()[1])
            j+=1

    volumes = volumes[nequil:] # don't worry about the first 1000 energies, the system is equilbrating.
    avevol = np.mean(volumes)
    if 'volume' in mapping:
        Lsize[i] = avevol**(1.0/3.0)  # scaling factor to map between the trajectories.  
                                      # We make the current trr more like the target (j) one.
    else:
        Lsize[i] = 1 # test without scaling

    f.write("%10.4f %10.4f +/- %6.4f %s\n" % (Lsize[i], avevol, np.std(volumes),d)) # just so we can see.

f.close()

original_volumes = np.zeros([nstates,npoints],float)
original_energies = np.zeros([nstates,npoints],float)

rerun_volumes = np.zeros([nstates,nstates,npoints]) 
rerun_energies = np.zeros([nstates,nstates,npoints]) 
lnJacobian = np.zeros([nstates,nstates])

for i,d in enumerate(dirnames):

    # now, generate new gros from the trr.
    tempdir = os.path.join(d,'tempdir')
    trrfile = os.path.join(d,'LJ_EM.trr')
    tprfile = os.path.join(d,'LJ_EM.tpr')
    runmdpfile = os.path.join(d,'LJequil.mdp')

    trrnojump = os.path.join(d,'LJ_EM_NOJUMP.trr')
    # now, we need to rewrite the coordinates so that we have no jumps.
    ps = subprocess.Popen(('echo', '0'), stdout=subprocess.PIPE) # get the energy and volume (hard coded)
    subprocess.call([gromacsbin,'trjconv','-f',trrfile,'-s',tprfile,'-o',trrnojump,'-pbc', 'nojump'], stdin=ps.stdout)

    if not os.path.isdir(tempdir):
        os.mkdir(tempdir)

    t = md.load(trrnojump,top=os.path.join(d,'LJ_EM.gro'))  # load the ith trajectory.
    t = t[nequil/nevery:]  # only take the equilibrated ones.

    if 'NVT' in dirtext:
        NVTvolume =  t[0].unitcell_volumes[0]

    for j in range(0,nstates):
        if 'harmonic' in mapping:
            temperature_scale = np.sqrt(Tpoints[j]/Tpoints[i])
        else:
            temperature_scale = 1
        tscale = t.slice(range(t.n_frames),copy=True) # force a copy to scale
        # first, scale them down to reduced coordinates using the box vectors.
        if 'NPT' in dirtext:
            for s, ts in enumerate(tscale):  # not sure why this doesn't work if we multiply in place.
                tscale.xyz[s] = np.array(tscale.xyz[s]*np.matrix(ts.unitcell_vectors)**-1)
        means = np.mean(tscale.xyz,axis=0)    
        divergences = (tscale.xyz-means)*temperature_scale  # calculate the deviations from the harmonic center, and scale up
        tscale.xyz = divergences+means  # add back the rescaled deviations from the harmonic center.
        # scale up/down coordinates and box by the trajectory, scale from configurations in i to configurations in j
        # assume isotropic expansion (for now), should only affect efficiency?
        tscale.unitcell_vectors *= Lsize[j]/Lsize[i]

        if 'NPT' in dirtext:
            for s, ts in enumerate(tscale):  # not sure why this doesn't work if we multiply in place.
                tscale.xyz[s] = np.array(tscale.xyz[s]*np.matrix(ts.unitcell_vectors))

        # save this trajectory    
        # This trajectory has beem mapped (simple distance scaling) from i to j 
        runprefix = 'LJ_PROD' + '_from_' + str(i) + '_to_' + str(j)
        savetrr = os.path.join(tempdir, runprefix + 'rerun.trr')
        savetpr = os.path.join(tempdir, runprefix + '.tpr')
        tscale.save(savetrr)

        # generate the new energies by running mdrun
        # open the mdp that was used, and make a new one
        # assume old TPR doesn't work, generate new one to run with rerun

        tmptpr = os.path.join(tempdir,'tmp.tpr')
        subprocess.call([gromacsbin,'grompp','-f',runmdpfile,'-c',os.path.join(d,'LJ_EM.gro'),'-o',tmptpr,'-p',os.path.join(d,'LJ.top'),'-maxwarn','3','-po',os.path.join(d,os.path.join(tempdir,'mdout.mdp'))])
        fname = os.path.join(tempdir, runprefix)
        subprocess.call([gromacsbin,'mdrun','-nt','1','-rerun',savetrr,'-s',tmptpr,'-deffnm',fname])
        tmpedr = os.path.join(tempdir,runprefix + '.edr')
        tmpxvg = os.path.join(tempdir,runprefix + '.xvg')
        ps = subprocess.Popen(('echo', '4','14'), stdout=subprocess.PIPE) # get the energy and volume (hard coded)
        subprocess.call([gromacsbin,'energy','-f',tmpedr,'-o',tmpxvg, '-dp'],stdin=ps.stdout)

        # now, read in the new energies.  These will be used for reweighting.
        rerun_energy_file = open(os.path.join(d,tmpxvg),'r')
        lines = rerun_energy_file.readlines()
        rerun_energy_file.close()
        n = 0
        for l in lines:
            if l[0] != '#' and l[0] != '@': 
                values = l.split()
                # these are the energies of E(T_{i->j}(x_i))
                # these energies are correct to within 0.01 kJ/mol or so (based on spot check of original xvg files)
                # compared to rerun .xvg files).
                rerun_energies[i,j,n] = float(values[1])   
                if 'NVT' in dirtext:
                    rerun_volumes[i,j,n] = NVTvolume
                else:
                    rerun_volumes[i,j,n] = float(values[2])
                n += 1
        if j==i:
            original_volumes[i,:] = rerun_volumes[i,i,:]
            original_energies[i,:] = rerun_energies[i,i,:]

    shutil.rmtree(tempdir)

for i in range(0,nstates):
    for j in range(0,nstates):
        betaTj = 1.0/(KB*Tpoints[j]) # beta = 1 / (K_bT)
        betaBj = betaTj*Pconv*Ppoints[j] # beta P  in correct units.

        # formula is beta E(x) -> Beta E_i(T_{j->i(x)} - ln J_T_{j->i}(x)
        # we take the configurations from state j, map them to state i, and compute the energies in state i.
        # we subtract the jacobian going from j to i
        # reduced coordinates formula (r,V)  -- Both are the same now.
        # \int V \int x exp(-beta U(V,x)) exp(-beta PV) dx dV 
        # change of variables to x=p V^(1/3) (p = reduced coordinates, range from from 0-1)
        # I = \int V \int x exp(-beta U(p V^(1/3))) (V-1)^N exp(-beta PV)  dp^(3N-3) dV   #N-1 from the COM DOF
        # I = \int V \int p exp -u(p,V) dp^(3N-3) dV
        #            where u(p,V) -[beta U(p V^(1/3)) + beta PV + N ln V] 
        # in this case, the jacobian is simply s, since p is not changed by transformation, only v. 
        # finally integral is of the form beta P \int exp(-u) drdV to be unitless, so u -> u - (ln beta P)
        # though perhaps it should use 1/V_0?  Hard to sday.  But appears not to matter that much.

        # Jacobian of the T transformation:  rho = rho_0 + (T/T)^(1/2)(rho-rho_0)
        #                                    dr = T[i]/T[j] dr - for each DOF.
        lnJacobian[i,j] = 0
        if integration == 'reduced':
            # this is the Jacobian of the transformation from i to j, since it divides by the length i, multiples by j. 
            if 'volume' in mapping:
                lnJacobian[i,j] += 3*np.log(Lsize[j]/Lsize[i]) # only change in V goes into jacobian.
            if 'harmonic' in mapping:
                lnJacobian[i,j] += 0.5*(3*(npermol-1))*np.log(Tpoints[j]/Tpoints[i])
            u_kln[i,j,:] = (betaTj*rerun_energies[i,j,:] + betaBj*rerun_volumes[i,j,:]) - (npermol-1)*np.log(rerun_volumes[i,j,:]) - lnJacobian[i,j]
        elif integration == 'direct':
            # nonreduced coordinate formula (not sure on complete DOF counting, but should match above)
            if 'volume' in mapping:
                lnJacobian[i,j] += (3*npermol)*np.log(Lsize[j]/Lsize[i])
            if 'harmonic' in mapping:
                lnJacobian[i,j] += 0.5*(3*(npermol-1))*np.log(Tpoints[j]/Tpoints[i])
            u_kln[i,j,:] = (betaTj*rerun_energies[i,j,:] + betaBj*rerun_volumes[i,j,:]) - lnJacobian[i,j]
        else:
            print "No formula!"
#scatterplot
if plot:
    import matplotlib.pyplot as plt
    c = ['r','b','g','y','c']
    for i in range(0,nstates):
        for j in range(0,nstates):
            # plot all of the configurations mapped onto i
            plt.scatter(rerun_volumes[j,i,:],rerun_energies[j,i,:],c=c[j],alpha=0.3,marker='o')
            plt.scatter(original_volumes[j,:],original_energies[j,:],c=c[j],alpha=0.3,marker='v')
        plt.show()
        fname = 'graph' + str(i) + 'to' + str(j) + '.pdf'
        plt.savefig(fname)

N_k = npoints*np.ones([nstates],int)

mbar = pymbar.MBAR(u_kln,N_k,relative_tolerance=1.0e-10,verbose=True)
(Delta_f_ij_estimated, dDelta_f_ij_estimated) = mbar.getFreeEnergyDifferences()
print Delta_f_ij_estimated
print dDelta_f_ij_estimated

# check these in the case of two.
# try exponential averaging, to see if there is a difference. Seems to work.

if len(dirnames) == 2:
    wf = -(u_kln[0,1:,]-u_kln[0,0,:])
    (df_forward,ddf_forward) = pymbar.EXP(wf)
    print('EXP forward %10.4f +/- %7.4f' % (df_forward, ddf_forward))
    wr = -(u_kln[1,1:,]-u_kln[1,0,:])
    (df_rev, ddf_rev) = pymbar.EXP(wr)
    print('EXP reverse %10.4f +/- %7.4f' % (df_rev, ddf_rev))
    
    pdb.set_trace()
    
    (df_bar, ddf_bar) = pymbar.BAR(-wf,wr)
    print('BAR reverse %10.4f +/- %7.4f' % (-df_bar, ddf_bar))
    
    # plots the overlap in energy betwen two.  Looks good!
    if plot:
        plt.clf()
        plt.hist(wf.T, facecolor='red')
        plt.hist(wr.T, facecolor='blue')
        plt.show()

