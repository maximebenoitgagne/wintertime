from numpy import prod, zeros

def split_UV_cub(u3d,v3d,kad=1):
    """
        u3d[...,nc,6*nc]
        v3d[...,nc,6*nc]
        
        split 2d/3d arrays u,v to 3d x 6 faces, and add extra cols/rows:
        kad==0:  returns u[6,...,nc,nc], v[6,...,nc,nc]
        kad==1:  returns u[6,...,nc,nc+1], v[6,...,nc+1,nc]
        ---------------------------------------------
        adapted by jahn@mit.edu from jmc@ocean.mit.edu 2009
    """

    dims = list(u3d.shape)
    if dims[-1] != 6*dims[-2]:
        print ' ERROR in split_UV_cub: 1st array has the wrong shape !'
        print ' dimensions:', dims
        return None, None

    dims = list(v3d.shape)
    ndim = len(dims)
    if dims[-1] != 6*dims[-2]:
        print ' ERROR in split_UV_cub: 2nd array has the wrong shape !'
        print ' dimensions:', dims
        return None, None

    nc,nx = dims[-2:]
    nr = int(prod(dims[:-2]))
    ncp = nc + 1
    n2p = nc + 2

    u3d = u3d.reshape((nr,nc,6,nc)).transpose((2,0,1,3))
    v3d = v3d.reshape((nr,nc,6,nc)).transpose((2,0,1,3))

    if kad == 0:

        # put extra dims back in
        u6t = u3d.reshape([6] + dims[:-2] + [nc, nc])
        v6t = v3d.reshape([6] + dims[:-2] + [nc, nc])

        return u6t, v6t

    elif kad == 1:

        #- split on to 6 faces with overlap in i+1 for u and j+1 for v :
        u6t = zeros((6,nr,nc,ncp))
        v6t = zeros((6,nr,ncp,nc))
        u6t[:,:,:,:nc] = u3d[:,:,:,:]
        v6t[:,:,:nc,:] = v3d[:,:,:,:]
       
        u6t[0,:,:,nc] = u3d[1,:,:,0]
        u6t[1,:,:,nc] = v3d[3,:,0,nc-1::-1]
        u6t[2,:,:,nc] = u3d[3,:,:,0]
        u6t[3,:,:,nc] = v3d[5,:,0,nc-1::-1]
        u6t[4,:,:,nc] = u3d[5,:,:,0]
        u6t[5,:,:,nc] = v3d[1,:,0,nc-1::-1]
       
        v6t[0,:,nc,:] = u3d[2,:,nc-1::-1,0]
        v6t[1,:,nc,:] = v3d[2,:,0,:]
        v6t[2,:,nc,:] = u3d[4,:,nc-1::-1,0]
        v6t[3,:,nc,:] = v3d[4,:,0,:]
        v6t[4,:,nc,:] = u3d[0,:,nc-1::-1,0]
        v6t[5,:,nc,:] = v3d[0,:,0,:]

        #- restore the right shape:
        u6t = u6t.reshape([6] + dims[:-2] + [nc, ncp])
        v6t = v6t.reshape([6] + dims[:-2] + [ncp, nc])

        return u6t, v6t

    else:

        print 'kad =',kad,'not supported'
        return None, None


