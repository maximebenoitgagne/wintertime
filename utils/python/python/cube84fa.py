from h5fa import FaFile

h5dir = '/net/eofe-data002/micklab002/jahn/h5/cube84/'

def mitgrid(*args, **kwargs):
    return FaFile(h5dir + 'fagrid.h5', *args, **kwargs)

