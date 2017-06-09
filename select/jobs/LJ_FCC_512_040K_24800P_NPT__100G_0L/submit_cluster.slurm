#! /bin/bash


#SBATCH --time=0-01:00:00
#SBATCH --output=logfile
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 16
#SBATCH --partition RM-shared

# submit a single SLURM job for a free energy calc
#module load gromacs/5.0.2-sse
export GRO_LOC="/projects/nasc5274/gromacs/gromacs/build/bin/gmx"
export SCRIPT_LOC="/home/nasc5274/scripts"


export THREADINFO="-nt 2 "

export OMP_NUM_THREADS=1

sleep 1

#Change the job status to 'SUBMITTED'
echo "SUBMITTED" > jobstatus.txt

#EQUILIBRATION

#gmx_d grompp -f equilibration.mdp -c pre_EQ.gro -r restraint.gro -p topology.top -o EQ.tpr  -maxwarn 10
#gmx_d mdrun ${THREADINFO} -v -deffnm EQ -dhdl dhdl_EQ
grompp_d -f LJequil.mdp -c initial.gro -r initial.gro -p LJ.top -o LJ_EM.tpr -maxwarn 10
mdrun_d ${THREADINFO} -v -deffnm LJ_EM


sleep 1

#EXTRACT THE FINAL FRAME FROM THE OUTPUT XTC FILE AND USE IT AS THE FINAL EQUILIBRATION FRAME

#echo "Total-Energy" | gmx g_energy -f EQ.edr -o time.xvg
#finaltime=$(tail -1 time.xvg | awk '{print $1}')
#rm EQ.gro
#echo 0 | gmx trjconv -f EQ.xtc -s EQ.tpr -o EQ.gro -pbc whole -ndec 12 -dump $finaltime

echo "Total-Energy" | g_energy_d -f LJ_EM.edr -o time.xvg
finaltime=$(tail -1 time.xvg | awk '{print $1}')
#rm LJ_EM.gro
echo 0 | trjconv_d -f LJ_EM.xtc -s LJ_EM.tpr -o LJ_EM.gro -pbc whole -ndec 12 -dump $finaltime





sleep 1

#PRODUCTION

#gmx grompp -f production.mdp -c EQ.gro -r restraint.gro -p topology.top -o PROD.tpr  -maxwarn 10
#gmx mdrun ${THREADINFO} -v -deffnm PROD -dhdl dhdl_PROD
grompp_d -f LJprod.mdp -c LJ_EM.gro -r initial.gro -p LJ.top -o LJ_PROD.tpr -maxwarn 10
mdrun_d ${THREADINFO} -v -deffnm LJ_PROD



sleep 1

#EXTRACT THE FINAL FRAME FROM THE OUTPUT TRR FILE AND USE IT AS THE FINAL PRODUCTION FRAME

#echo "Total-Energy" | gmx g_energy -f PROD.edr -o time.xvg
#finaltime=$(tail -1 time.xvg | awk '{print $1}')
#rm PROD.gro
#echo 0 | gmx trjconv -f PROD.trr -s PROD.tpr -o PROD.gro -pbc whole -ndec 12 -dump $finaltime
echo "Total-Energy" | g_energy_d -f LJ_PROD.edr -o time.xvg
finaltime=$(tail -1 time.xvg | awk '{print $1}')
#rm LJ_PROD.gro
echo 0 | trjconv_d -f LJ_PROD.trr -s LJ_PROD.tpr -o LJ_PROD.gro -pbc whole -ndec 12 -dump $finaltime
echo 0 | trjconv_d -f LJ_PROD.trr -s LJ_PROD.tpr -o LJ_PROD.pdb -pbc nojuump -ndec 8

sleep 1

#CALCULATE THE CONFIGURATION ENERGIES AND ENSEMBLE AVERAGES
crunchjobenergy

sleep 1


#RELAX THE BENZENES TO CREATE THE RESTRAINT FILES
#./relax_benzene.sh

#REWEIGHT THE JOB INTO THE OTHER POTENTIALS
#${SCRIPT_LOC}/reweightjobgromacs -s gromacs -u "oplsaafakea"
#${SCRIPT_LOC}/convertjobtinker -s gromacs
#rm -rf TinkerXYZ

#RELAX THE MOLECULE AND ENERGY MINIMIZE THE INTRAMOLECULAR INTERACTIONS TO CREATE THE RESTRAINT FILES
#./relax_molecule.sh



#DELETE THE LOGFILE IF THE JOB FINISHED CORRECTLY
if [ -f PROD.log ] && [ "$(tail -1 PROD.log | awk '{print $1}')" == "Finished" ]; then
    rm logfile
fi



echo "FINISHED" > jobstatus.txt
# print end time
echo
echo "Job Ended at `date`"
echo "###################################################################"
                                                                                                                                                                                    73,1          Bot



