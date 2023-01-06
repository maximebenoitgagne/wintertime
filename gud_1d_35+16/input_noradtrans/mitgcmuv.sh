#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=512M
#SBATCH --account=def-fmaps
#SBATCH --job-name=mitgcmuv
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.benoit-gagne@takuvik.ulaval.ca
module load netcdf-fortran/4.4.4
./mitgcmuv
module purge
