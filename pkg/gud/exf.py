__all__ = ['exfflds']

# add here new fields to be read in through namelist gud_forcing_params
# first entry is name of the variable to read into
# second is the prefix for runtime paramaters such as PARfile
exfflds = [
    ['surfPAR', 'PAR'],
    ['inputFe', 'iron'],
    ['iceFrac', 'ice'],
    ['windSpeed', 'wind'],
    ['atmospCO2', 'pCO2']
    ]

