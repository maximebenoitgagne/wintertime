#__all__ = ['hr']

physdirs = {'LR'  : '/data/jahn/gyre/input/B1VD/B1VD_mon',
            'HR'  : '/data/jahn/gyre/input/B54V3onB9/offline',
            '3D-e': '/data/jahn/gyre/input/B54V3onB9/offline',
            '3D-m': '/data/jahn/gyre/input/B54V3onB9/offline',
            '1D'  : '/data/jahn/gyre/input/B54V3onB9/offline',
            '0D'  : '/data/jahn/gyre/input/B54V3onB9/offline',
           }
diagdirs = {'LR'  : '/data/jahn/gyre/run/B1/monod/run.002/diags',
            'HR'  : '/data/jahn/gyre/run/B9/monod/run.017/diags',
            '3D-e': '/data/jahn/gyre/run/B9/monod/run.020/diags',
            '3D-m': '/data/jahn/gyre/run/B9/monod/run.021/diags',
            '1D'  : '/data/jahn/gyre/run/B9/monod/run.022/diags',
            '0D'  : '/data/jahn/gyre/run/B9/monod/run.023/diags',
           }

#import hr, runLR, run3De, run3Dm, run1D, run0D
#
#mon = {'HR'  : hr.mon,
#       'LR'  : runLR.mon,
#       '3D-e': run3De.mon,
#       '3D-m': run3Dm.mon,
#       '1D'  : run1D.mon,
#       '0D'  : run0D.mon,
#      }
