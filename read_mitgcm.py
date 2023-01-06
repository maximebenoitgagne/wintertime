#!/usr/bin/env python3

"""read data from an output file of the biogeochemical component of MITgcm
"""

# author: Maxime Benoit-Gagne - ULaval - Canada.
# date of creation: July 28, 2022.
#
# Python from Anaconda$ python
# Python 3.7.12 | packaged by conda-forge | (default, Oct 26 2021, 05:57:50) 
# [Clang 11.1.0 ] on darwin
# Type "help", "copyright", "credits" or "license" for more information.

########### Importing modules ###########

import numpy as np

import netcdf_tools

########### function ###########

"""
get Chl a concentration for all types

Args:
    infile(str):
        The netCDF4 file for Chl a from a simulation by the biogeochemical 
        component of MITgcm.

Returns:
    array2d_idepth_iT_chlfull(array-like):
        Array of 2 dimensions.
        The first dimension is the indices of the depths.
        The second dimension is the indices of the time steps.
        The values are the Chl a concentration (mg Chl m^-3).
"""
def get_array2d_idepth_iT_chlfull(infile):
    array2d_idepth_iT_prochlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC70').squeeze().transpose()
    array2d_idepth_iT_prochlfull[-1,:]=np.nan

    array2d_idepth_iT_synchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC71').squeeze().transpose()
    array2d_idepth_iT_synchlfull[-1,:]=np.nan

    array2d_idepth_iT_smalleuk1umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC72').squeeze().transpose()
    array2d_idepth_iT_smalleuk1umchlfull[-1,:]=np.nan

    array2d_idepth_iT_smalleuk2umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC73').squeeze().transpose()
    array2d_idepth_iT_smalleuk2umchlfull[-1,:]=np.nan

    array2d_idepth_iT_smalleukchlfull\
    =array2d_idepth_iT_smalleuk1umchlfull+array2d_idepth_iT_smalleuk2umchlfull

    array2d_idepth_iT_cocco3umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC74').squeeze().transpose()
    array2d_idepth_iT_cocco3umchlfull[-1,:]=np.nan
    array2d_idepth_iT_cocco4umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC75').squeeze().transpose()
    array2d_idepth_iT_cocco4umchlfull[-1,:]=np.nan
    array2d_idepth_iT_cocco7umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC76').squeeze().transpose()
    array2d_idepth_iT_cocco7umchlfull[-1,:]=np.nan
    array2d_idepth_iT_cocco10umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC77').squeeze().transpose()
    array2d_idepth_iT_cocco10umchlfull[-1,:]=np.nan
    array2d_idepth_iT_cocco15umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC78').squeeze().transpose()
    array2d_idepth_iT_cocco15umchlfull[-1,:]=np.nan

    array2d_idepth_iT_coccochlfull\
    =array2d_idepth_iT_cocco3umchlfull+array2d_idepth_iT_cocco4umchlfull\
    +array2d_idepth_iT_cocco7umchlfull+array2d_idepth_iT_cocco10umchlfull\
    +array2d_idepth_iT_cocco15umchlfull

    array2d_idepth_iT_diazo3umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC79').squeeze().transpose()
    array2d_idepth_iT_diazo3umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diazo4umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC80').squeeze().transpose()
    array2d_idepth_iT_diazo4umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diazo7umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC81').squeeze().transpose()
    array2d_idepth_iT_diazo7umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diazo10umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC82').squeeze().transpose()
    array2d_idepth_iT_diazo10umchlfull[-1,:]=np.nan

    array2d_idepth_iT_diazochlfull\
    =array2d_idepth_iT_diazo3umchlfull+array2d_idepth_iT_diazo4umchlfull\
    +array2d_idepth_iT_diazo7umchlfull+array2d_idepth_iT_diazo10umchlfull

    array2d_idepth_iT_trichlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC83').squeeze().transpose()
    array2d_idepth_iT_trichlfull[-1,:]=np.nan

    array2d_idepth_iT_diatom7umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC84').squeeze().transpose()
    array2d_idepth_iT_diatom7umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom10umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC85').squeeze().transpose()
    array2d_idepth_iT_diatom10umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom15umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC86').squeeze().transpose()
    array2d_idepth_iT_diatom15umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom22umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC87').squeeze().transpose()
    array2d_idepth_iT_diatom22umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom32umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC88').squeeze().transpose()
    array2d_idepth_iT_diatom32umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom47umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC89').squeeze().transpose()
    array2d_idepth_iT_diatom47umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom70umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC90').squeeze().transpose()
    array2d_idepth_iT_diatom70umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom104umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC91').squeeze().transpose()
    array2d_idepth_iT_diatom104umchlfull[-1,:]=np.nan
    array2d_idepth_iT_diatom154umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC92').squeeze().transpose()
    array2d_idepth_iT_diatom154umchlfull[-1,:]=np.nan

    array2d_idepth_iT_diatomchlfull\
    =array2d_idepth_iT_diatom7umchlfull+array2d_idepth_iT_diatom10umchlfull\
    +array2d_idepth_iT_diatom15umchlfull+array2d_idepth_iT_diatom22umchlfull\
    +array2d_idepth_iT_diatom32umchlfull+array2d_idepth_iT_diatom47umchlfull\
    +array2d_idepth_iT_diatom70umchlfull+array2d_idepth_iT_diatom104umchlfull\
    +array2d_idepth_iT_diatom154umchlfull

    array2d_idepth_iT_largeeuk7umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC93').squeeze().transpose()
    array2d_idepth_iT_largeeuk7umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk10umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC94').squeeze().transpose()
    array2d_idepth_iT_largeeuk10umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk15umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC95').squeeze().transpose()
    array2d_idepth_iT_largeeuk15umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk22umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC96').squeeze().transpose()
    array2d_idepth_iT_largeeuk22umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk32umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC97').squeeze().transpose()
    array2d_idepth_iT_largeeuk32umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk47umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC98').squeeze().transpose()
    array2d_idepth_iT_largeeuk47umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk70umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC99').squeeze().transpose()
    array2d_idepth_iT_largeeuk70umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk104umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC0a').squeeze().transpose()
    array2d_idepth_iT_largeeuk104umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk154umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC0b').squeeze().transpose()
    array2d_idepth_iT_largeeuk154umchlfull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk228umchlfull\
    =netcdf_tools.read_netcdf(infile, 'TRAC0c').squeeze().transpose()
    array2d_idepth_iT_largeeuk228umchlfull[-1,:]=np.nan

    array2d_idepth_iT_largeeukchlfull\
    =array2d_idepth_iT_largeeuk7umchlfull\
    +array2d_idepth_iT_largeeuk10umchlfull\
    +array2d_idepth_iT_largeeuk15umchlfull\
    +array2d_idepth_iT_largeeuk22umchlfull\
    +array2d_idepth_iT_largeeuk32umchlfull\
    +array2d_idepth_iT_largeeuk47umchlfull\
    +array2d_idepth_iT_largeeuk70umchlfull\
    +array2d_idepth_iT_largeeuk104umchlfull\
    +array2d_idepth_iT_largeeuk154umchlfull\
    +array2d_idepth_iT_largeeuk228umchlfull

    array2d_idepth_iT_chlfull\
    =array2d_idepth_iT_prochlfull+array2d_idepth_iT_synchlfull\
    +array2d_idepth_iT_smalleukchlfull+array2d_idepth_iT_coccochlfull\
    +array2d_idepth_iT_diazochlfull+array2d_idepth_iT_trichlfull\
    +array2d_idepth_iT_diatomchlfull+array2d_idepth_iT_largeeukchlfull

    return array2d_idepth_iT_chlfull

########### function ###########

"""
get phytoplankton biomass concentration for all types

Args:
    infile(str):
        The netCDF4 file for biomass from a simulation by the biogeochemical 
        component of MITgcm.

Returns:
    array2d_idepth_iT_phytobiofull(array-like):
        Array of 2 dimensions.
        The first dimension is the indices of the depths.
        The second dimension is the indices of the time steps.
        The values are the phytoplankton biomass concentration (mg C m^-3).
"""
def get_array2d_idepth_iT_phytobiofull(infile):
    molarmassC=12.0107 # g C (mol C)^-1

    array2d_idepth_iT_probiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC21').squeeze().transpose()
    array2d_idepth_iT_probiofull[-1,:]=np.nan
    # mmol C m^-3 -> g C m^-3
    array2d_idepth_iT_probiofull\
    =array2d_idepth_iT_probiofull*molarmassC/1e3

    array2d_idepth_iT_synbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC22').squeeze().transpose()
    array2d_idepth_iT_synbiofull[-1,:]=np.nan
    array2d_idepth_iT_synbiofull\
    =array2d_idepth_iT_synbiofull*molarmassC/1e3

    array2d_idepth_iT_smalleuk1umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC23').squeeze().transpose()
    array2d_idepth_iT_smalleuk1umbiofull[-1,:]=np.nan
    array2d_idepth_iT_smalleuk1umbiofull\
    =array2d_idepth_iT_smalleuk1umbiofull*molarmassC/1e3

    array2d_idepth_iT_smalleuk2umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC24').squeeze().transpose()
    array2d_idepth_iT_smalleuk2umbiofull[-1,:]=np.nan
    array2d_idepth_iT_smalleuk2umbiofull\
    =array2d_idepth_iT_smalleuk2umbiofull*molarmassC/1e3

    array2d_idepth_iT_smalleukbiofull\
    =array2d_idepth_iT_smalleuk1umbiofull+array2d_idepth_iT_smalleuk2umbiofull

    array2d_idepth_iT_cocco3umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC25').squeeze().transpose()
    array2d_idepth_iT_cocco3umbiofull[-1,:]=np.nan
    array2d_idepth_iT_cocco3umbiofull\
    =array2d_idepth_iT_cocco3umbiofull*molarmassC/1e3
    array2d_idepth_iT_cocco4umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC26').squeeze().transpose()
    array2d_idepth_iT_cocco4umbiofull[-1,:]=np.nan
    array2d_idepth_iT_cocco4umbiofull\
    =array2d_idepth_iT_cocco4umbiofull*molarmassC/1e3
    array2d_idepth_iT_cocco7umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC27').squeeze().transpose()
    array2d_idepth_iT_cocco7umbiofull[-1,:]=np.nan
    array2d_idepth_iT_cocco7umbiofull\
    =array2d_idepth_iT_cocco7umbiofull*molarmassC/1e3
    array2d_idepth_iT_cocco10umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC28').squeeze().transpose()
    array2d_idepth_iT_cocco10umbiofull[-1,:]=np.nan
    array2d_idepth_iT_cocco10umbiofull\
    =array2d_idepth_iT_cocco10umbiofull*molarmassC/1e3
    array2d_idepth_iT_cocco15umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC29').squeeze().transpose()
    array2d_idepth_iT_cocco15umbiofull[-1,:]=np.nan
    array2d_idepth_iT_cocco15umbiofull\
    =array2d_idepth_iT_cocco15umbiofull*molarmassC/1e3

    array2d_idepth_iT_modcoccobiofull\
    =array2d_idepth_iT_cocco3umbiofull+array2d_idepth_iT_cocco4umbiofull\
    +array2d_idepth_iT_cocco7umbiofull+array2d_idepth_iT_cocco10umbiofull\
    +array2d_idepth_iT_cocco15umbiofull

    array2d_idepth_iT_diazo3umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC30').squeeze().transpose()
    array2d_idepth_iT_diazo3umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diazo3umbiofull\
    =array2d_idepth_iT_diazo3umbiofull*molarmassC/1e3
    array2d_idepth_iT_diazo4umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC31').squeeze().transpose()
    array2d_idepth_iT_diazo4umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diazo4umbiofull\
    =array2d_idepth_iT_diazo4umbiofull*molarmassC/1e3
    array2d_idepth_iT_diazo7umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC32').squeeze().transpose()
    array2d_idepth_iT_diazo7umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diazo7umbiofull\
    =array2d_idepth_iT_diazo7umbiofull*molarmassC/1e3
    array2d_idepth_iT_diazo10umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC33').squeeze().transpose()
    array2d_idepth_iT_diazo10umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diazo10umbiofull\
    =array2d_idepth_iT_diazo10umbiofull*molarmassC/1e3

    array2d_idepth_iT_diazobiofull\
    =array2d_idepth_iT_diazo3umbiofull+array2d_idepth_iT_diazo4umbiofull\
    +array2d_idepth_iT_diazo7umbiofull+array2d_idepth_iT_diazo10umbiofull

    array2d_idepth_iT_tribiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC34').squeeze().transpose()
    array2d_idepth_iT_tribiofull[-1,:]=np.nan
    array2d_idepth_iT_tribiofull\
    =array2d_idepth_iT_tribiofull*molarmassC/1e3

    array2d_idepth_iT_diatom7umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC35').squeeze().transpose()
    array2d_idepth_iT_diatom7umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom7umbiofull\
    =array2d_idepth_iT_diatom7umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom10umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC36').squeeze().transpose()
    array2d_idepth_iT_diatom10umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom10umbiofull\
    =array2d_idepth_iT_diatom10umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom15umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC37').squeeze().transpose()
    array2d_idepth_iT_diatom15umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom15umbiofull\
    =array2d_idepth_iT_diatom15umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom22umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC38').squeeze().transpose()
    array2d_idepth_iT_diatom22umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom22umbiofull\
    =array2d_idepth_iT_diatom22umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom32umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC39').squeeze().transpose()
    array2d_idepth_iT_diatom32umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom32umbiofull\
    =array2d_idepth_iT_diatom32umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom47umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC40').squeeze().transpose()
    array2d_idepth_iT_diatom47umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom47umbiofull\
    =array2d_idepth_iT_diatom47umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom70umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC41').squeeze().transpose()
    array2d_idepth_iT_diatom70umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom70umbiofull\
    =array2d_idepth_iT_diatom70umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom104umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC42').squeeze().transpose()
    array2d_idepth_iT_diatom104umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom104umbiofull\
    =array2d_idepth_iT_diatom104umbiofull*molarmassC/1e3
    array2d_idepth_iT_diatom154umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC43').squeeze().transpose()
    array2d_idepth_iT_diatom154umbiofull[-1,:]=np.nan
    array2d_idepth_iT_diatom154umbiofull\
    =array2d_idepth_iT_diatom154umbiofull*molarmassC/1e3

    array2d_idepth_iT_moddiatombiofull\
    =array2d_idepth_iT_diatom7umbiofull+array2d_idepth_iT_diatom10umbiofull\
    +array2d_idepth_iT_diatom15umbiofull+array2d_idepth_iT_diatom22umbiofull\
    +array2d_idepth_iT_diatom32umbiofull+array2d_idepth_iT_diatom47umbiofull\
    +array2d_idepth_iT_diatom70umbiofull+array2d_idepth_iT_diatom104umbiofull\
    +array2d_idepth_iT_diatom154umbiofull

    array2d_idepth_iT_largeeuk7umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC44').squeeze().transpose()
    array2d_idepth_iT_largeeuk7umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk7umbiofull\
    =array2d_idepth_iT_largeeuk7umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk10umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC45').squeeze().transpose()
    array2d_idepth_iT_largeeuk10umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk10umbiofull\
    =array2d_idepth_iT_largeeuk10umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk15umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC46').squeeze().transpose()
    array2d_idepth_iT_largeeuk15umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk15umbiofull\
    =array2d_idepth_iT_largeeuk15umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk22umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC47').squeeze().transpose()
    array2d_idepth_iT_largeeuk22umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk22umbiofull\
    =array2d_idepth_iT_largeeuk22umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk32umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC48').squeeze().transpose()
    array2d_idepth_iT_largeeuk32umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk32umbiofull\
    =array2d_idepth_iT_largeeuk32umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk47umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC49').squeeze().transpose()
    array2d_idepth_iT_largeeuk47umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk47umbiofull\
    =array2d_idepth_iT_largeeuk47umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk70umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC50').squeeze().transpose()
    array2d_idepth_iT_largeeuk70umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk70umbiofull\
    =array2d_idepth_iT_largeeuk70umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk104umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC51').squeeze().transpose()
    array2d_idepth_iT_largeeuk104umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk104umbiofull\
    =array2d_idepth_iT_largeeuk104umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk154umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC52').squeeze().transpose()
    array2d_idepth_iT_largeeuk154umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk154umbiofull\
    =array2d_idepth_iT_largeeuk154umbiofull*molarmassC/1e3
    array2d_idepth_iT_largeeuk228umbiofull\
    =netcdf_tools.read_netcdf(infile, 'TRAC53').squeeze().transpose()
    array2d_idepth_iT_largeeuk228umbiofull[-1,:]=np.nan
    array2d_idepth_iT_largeeuk228umbiofull\
    =array2d_idepth_iT_largeeuk228umbiofull*molarmassC/1e3

    array2d_idepth_iT_moddinobiofull\
    =array2d_idepth_iT_largeeuk7umbiofull+array2d_idepth_iT_largeeuk10umbiofull\
    +array2d_idepth_iT_largeeuk15umbiofull+array2d_idepth_iT_largeeuk22umbiofull\
    +array2d_idepth_iT_largeeuk32umbiofull+array2d_idepth_iT_largeeuk47umbiofull\
    +array2d_idepth_iT_largeeuk70umbiofull+array2d_idepth_iT_largeeuk104umbiofull\
    +array2d_idepth_iT_largeeuk154umbiofull+array2d_idepth_iT_largeeuk228umbiofull

    array2d_idepth_iT_modphytobiofull\
    =array2d_idepth_iT_probiofull+array2d_idepth_iT_synbiofull\
    +array2d_idepth_iT_smalleukbiofull+array2d_idepth_iT_modcoccobiofull\
    +array2d_idepth_iT_diazobiofull+array2d_idepth_iT_tribiofull\
    +array2d_idepth_iT_moddiatombiofull+array2d_idepth_iT_moddinobiofull

    return array2d_idepth_iT_modphytobiofull
