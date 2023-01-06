#!/usr/bin/env python
import matplotlib
from matplotlib.pyplot import text, gca, draw

def mytext(x,y,s,**kwargs):
    """
        like text, but if kwarg 'model' is given, place s where
        the text in kwargs['model'] would start:
        1. call text with s replaced by kwargs['model'] and remove the result
        2. call text with s, but baseline-left justify at bottom-left corner 
           of result of 1.
        Note: this works best if the model text does not have descenders.
    """
    # we take care of this one
    model = kwargs.pop('model', None)
    if model:
        th = text(x,y,model,**kwargs)
        draw()
        x0,y0,w,h = th.get_window_extent().bounds
        gca().texts.remove(th)
        x = x0
        y = y0
        kwargs['transform'] = matplotlib.transforms.IdentityTransform()
        kwargs['horizontalalignment'] = 'left'
        kwargs['verticalalignment'] = 'baseline'
#        print x,y,kwargs
    return text(x,y,s,**kwargs)

