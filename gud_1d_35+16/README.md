I present first a general procedure to compile and run on any UNIX system.
Then, I present the exact procedure I used on a supercomputer of the Digital Research Alliance of Canada with the SLURM workload manager.

# General procedure to compile and run on any UNIX system

## to compile

Modifications for the code are in directory `code_noradtrans`.

go to directory `bin`

```
../../tools/genmake2 -mods=../code_noradtrans/
make depend
make
```

This will produce an executable `mitgcmuv` in the `bin` directory.

## to run

From the gud_1d_35+16 directory:

```
cp -r input_noradtrans run_noradtrans
cp bin/mitgcmuv run_noradtrans
cd run_noradtrans
nohup ./mitgcmuv > output.txt 2> output.err &
```

output will be in `diags_***` directory

# Exact procedure I used to compile and run on a supercomputer of the Digital Research Alliance of Canada with SLURM

Replace mysimulation with a meaningful name.

## load modules

```
module load nixpkgs/16.09 
module load gcc/5.4.0
module load netcdf-fortran/4.4.4
```

## to compile

Modifications for the code are in the directory `code_noradtrans`.

I call the `bin` directory the directory `build_code_mysimulation` here.

```
cp -r code_noradtrans code_mysimulation
mkdir build_code_mysimulation
cd build_code_mysimulation
rm -f *
../../tools/genmake2 -mods=../code_mysimulation 
make depend
make -j 12 
cd ..
```

This will produce an executable `mitgcmuv` in the `build_code_mysimulation` directory.

## to run

```
cp -r input_noradtrans run_mysimulation
cp build_code_mysimulation/mitgcmuv run_mysimulation/mitgcmuv_mysimulation
cd run_mysimulation
ln -s mitgcmuv_mysimulation mitgcmuv
./prepare_run
sbatch mitgcmuv.sh
cd ..
```

output will be in `diags_***` directory
