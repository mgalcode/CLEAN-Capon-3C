def plt_hist(cln_hist,cln_iter):
    import numpy as np
    import matplotlib.pyplot as plt
    
    baz = np.zeros([3,cln_iter+1])
    vel = np.zeros([3,cln_iter+1])
    idd = 0
    for i in cln_hist:
        for j in range(3):
            x =  np.arctan2(i[j,0],i[j,1])/np.pi*180
            if x<0: x+=360
            baz[j,idd] = x
            vel[j,idd] = 111.19/np.sqrt(i[j,0]**2 + i[j,1]**2)
        idd += 1

    fig = plt.figure(figsize=(25, 6))
    ax1=fig.add_subplot(131)
    im = ax1.scatter(range(cln_iter+1), baz[0], s=50 ,c=vel[0], alpha=1,cmap='bwr')
    cbar = plt.colorbar(im)
    cbar.set_label('velocity [km/s]')
    ax2=fig.add_subplot(132)
    im = ax2.scatter(range(cln_iter+1), baz[1], s=50 ,c=vel[1], alpha=1,cmap='bwr')
    cbar = plt.colorbar(im)
    cbar.set_label('velocity [km/s]')
    ax3=fig.add_subplot(133)
    im = ax3.scatter(range(cln_iter+1), baz[2], s=50 ,c=vel[2], alpha=1,cmap='bwr')
    cbar=plt.colorbar(im)
    cbar.set_label('velocity [km/s]')
    ax1.set_ylim([0,360])
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('backazimuth [deg]')
    ax2.set_ylim([0,360])
    ax2.set_xlabel('iteration')
    ax3.set_ylim([0,360])
    ax3.set_xlabel('iteration')



def refinment_fk(ref_fk,sxopt,syopt,nk,rx,ry,nr,icsdm,freq,sinc,k):
    import numpy as np
    import scipy as sp
    from subroutine_CLEAN_3c import rot

    steer = np.zeros((3,3*nr),dtype=complex)
    norm = 1/np.sqrt(nr)
    for i in range(-1,2):
        kx=-2*np.pi*freq*(sxopt + sinc*i)
        for j in range(-1,2):
            ky=-2*np.pi*freq*(syopt + sinc*j)
            steer[0,:nr]=np.exp(1j*(kx*(rx[0]-rx)+ky*(ry[0]-ry)))*norm
            steer[1,nr:2*nr] = steer[0,:nr]
            steer[2,2*nr:] = steer[0,:nr]

            theta = np.arctan2(kx,ky)
            xres = steer.conj().dot(icsdm).dot(steer.T)
            u,v = np.linalg.eigh(xres)
            uid = u.argsort()[::-1]
            i0 = uid[0]
            i1 = uid[1]
            z1   = v[0,i0]
            z2   = v[0,i1]
            bh11 = v[1,i0] 
            bh12 = v[1,i1] 
            bh21 = v[2,i0] 
            bh22 = v[2,i1] 
            r1,t1 = rot(bh11,bh21,theta)
            r2,t2 = rot(bh12,bh22,theta)
            if k == 0:
                ref_fk[i+1,j+1] =  u[i0].real * np.abs(z1)**2 +  u[i1].real * np.abs(z2)**2 #+  u[i2].real * np.abs(z3)**2 
            if k == 1:
                ref_fk[i+1,j+1] =  u[i0].real * np.abs(r1)**2 +  u[i1].real * np.abs(r2)**2 #+  u[i2].real * np.abs(r3)**2
            if k == 2:
                ref_fk[i+1,j+1] =  u[i0].real * np.abs(t1)**2 +  u[i1].real * np.abs(t2)**2 #+  u[i2].real * np.abs(t3)**2

    return ref_fk


def refine_max_fk(src_grd_ref,polariz,nk,nr,rx,ry,icsdm,max_c,smin,sinc,freq):
    import numpy as np
    for k in range(3):
        if max_c[k,0]!= smin and max_c[k,0]!=-smin and max_c[k,1]!= smin and max_c[k,1]!=-smin:
            i = np.round((max_c[k,0] - smin)/sinc).astype(np.int)
            j = np.round((max_c[k,1] - smin)/sinc).astype(np.int)
            ref_fk = polariz[i-1:i+2,j-1:j+2,k].copy()
            for ijk in range(src_grd_ref):
               xsinc = sinc/float((ijk+1)**2)
               ref_fk = refinment_fk(ref_fk,max_c[k,0],max_c[k,1],nk,rx,ry,nr,icsdm[k],freq,xsinc,k)
               i,j = np.unravel_index(ref_fk.argmax(), ref_fk.shape)
               i -= 1
               j -= 1
               max_c[k,0] += xsinc*i
               max_c[k,1] += xsinc*j
    return max_c


def refinment_Capon(ref_fk,sxopt,syopt,nk,rx,ry,nr,icsdm,freq,sinc,k):
    import numpy as np
    import scipy as sp
    from subroutine_CLEAN_3c import rot

    steer = np.zeros((3,3*nr),dtype=complex)
    norm = 1/np.sqrt(nr)
    for i in range(-1,2):
        kx=-2*np.pi*freq*(sxopt + sinc*i)
        for j in range(-1,2):
            ky=-2*np.pi*freq*(syopt + sinc*j)
            steer[0,:nr]=np.exp(1j*(kx*(rx[0]-rx)+ky*(ry[0]-ry)))*norm
            steer[1,nr:2*nr] = steer[0,:nr]
            steer[2,2*nr:] = steer[0,:nr]

            theta = np.arctan2(kx,ky)
            xres = steer.conj().dot(icsdm).dot(steer.T)
            u,v = np.linalg.eigh(xres)
            uid = u.argsort()
            i0 = uid[0]
            i1 = uid[1]
            z1   = v[0,i0]
            z2   = v[0,i1]
            bh11 = v[1,i0] 
            bh12 = v[1,i1] 
            bh21 = v[2,i0] 
            bh22 = v[2,i1] 
            r1,t1 = rot(bh11,bh21,theta)
            r2,t2 = rot(bh12,bh22,theta)
            if k == 0:
                ref_fk[i+1,j+1] = 1 / u[i0].real * np.abs(z1)**2 + 1 / u[i1].real * np.abs(z2)**2 #+ 1 / u[i2].real * np.abs(z3)**2 
            if k == 1:
                ref_fk[i+1,j+1] = 1 / u[i0].real * np.abs(r1)**2 + 1 / u[i1].real * np.abs(r2)**2 #+ 1 / u[i2].real * np.abs(r3)**2
            if k == 2:
                ref_fk[i+1,j+1] = 1 / u[i0].real * np.abs(t1)**2 + 1 / u[i1].real * np.abs(t2)**2 #+ 1 / u[i2].real * np.abs(t3)**2

    return ref_fk


def refine_max_Capon(src_grd_ref,polariz,nk,nr,rx,ry,icsdm,max_c,smin,sinc,freq):
    import numpy as np
    for k in range(3):
        if max_c[k,0]!= smin and max_c[k,0]!=-smin and max_c[k,1]!= smin and max_c[k,1]!=-smin:
            i = np.round((max_c[k,0] - smin)/sinc).astype(np.int)
            j = np.round((max_c[k,1] - smin)/sinc).astype(np.int)
            ref_fk = polariz[i-1:i+2,j-1:j+2,k].copy()
            for ijk in range(src_grd_ref):
               xsinc = sinc/float((ijk+1)**2)
               ref_fk = refinment_Capon(ref_fk,max_c[k,0],max_c[k,1],nk,rx,ry,nr,icsdm[k],freq,xsinc,k)
               i,j = np.unravel_index(ref_fk.argmax(), ref_fk.shape)
               i -= 1
               j -= 1
               max_c[k,0] += xsinc*i
               max_c[k,1] += xsinc*j
    return max_c


def get_rxy_sac(nr,st0):
    from subroutine_CLEAN_3c import grt
    import numpy as np
    rx = np.zeros(nr)
    ry = np.zeros(nr)
    for i in range(nr):
        decl,dist,az,baz = grt(st0[0].stats.sac.stla,st0[0].stats.sac.stlo,st0[i].stats.sac.stla,st0[i].stats.sac.stlo)
        rx[i] = decl*np.cos(0.017453*(90.0-az))
        ry[i] = decl*np.sin(0.017453*(90.0-az))
    return rx,ry


def remove_gain(st,st0,st1,nr,gain):
    for i in range(nr):
        st[i].data = st[i][:]/gain
        st0[i].data = st0[i][:]/gain
        st1[i].data = st1[i][:]/gain
    return st,st0,st1




def get_max(polariz,smin,sinc,cln):
    import numpy as np
    max_c = np.zeros((3,2))
    max_o = np.zeros(3)
    for k in range(3):
        i,j = np.unravel_index(polariz[:,:,k].argmax(), polariz[:,:,k].shape)
        max_c[k,0] = smin + i*sinc
        max_c[k,1] = smin + j*sinc


    if cln == 0:
        for k in range(3):
            max_o[k] = polariz[:,:,k].max()
    return max_c,max_o



def make_plot(Z,R,T,smin,smax,min_relative_pow,enhance_vis,inter_mode,area,std_g):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from subroutine_CLEAN_3c import gkern2

    kern = gkern2(kernlen=area,nsig=std_g)

    Z = signal.fftconvolve(Z,kern,mode='same')
    tmp = np.where(Z<=0)
    Z[tmp] = Z.max()*10**(min_relative_pow)

    R = signal.fftconvolve(R,kern,mode='same')
    tmp = np.where(R<=0)
    R[tmp] = R.max()*10**(min_relative_pow)

    T = signal.fftconvolve(T,kern,mode='same')
    tmp = np.where(T<=0)
    T[tmp] = T.max()*10**(min_relative_pow)


    fig = plt.figure(figsize=(15, 13))
    ax=fig.add_subplot(221)
    im = plt.imshow(10*np.log10((Z+R+T) / (Z+R+T).max()).T,vmin = min_relative_pow,extent=[smin,smax, smin, smax],origin='lower',cmap='gist_stern_r',interpolation=inter_mode)
    # circle=plt.Circle((0,0),0.3*111.19,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    # circle=plt.Circle((0,0),27,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    plt.title('CLEAN - Beamformer 3C ')
    ax.set_xlabel('East/West Slowness [s/deg]')
    ax.set_ylabel('North/South Slowness [s/deg]')
    cbar = plt.colorbar(im)
    cbar.set_label('relative power (dB)',rotation=270,labelpad=20)



    ax=fig.add_subplot(222)
    im = plt.imshow(10*np.log10(Z/Z.max()).T, extent=[smin,smax, smin,smax], vmin = min_relative_pow,vmax = 0,origin='lower',cmap='gist_stern_r',interpolation=inter_mode)
    # circle=plt.Circle((0,0),0.3*111.19,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    # circle=plt.Circle((0,0),27,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    plt.title('Vertical')
    ax.set_xlabel('East/West Slowness [s/deg]')
    ax.set_ylabel('North/South Slowness [s/deg]')
    cbar = plt.colorbar(im)
    cbar.set_label('relative power (dB)',rotation=270,labelpad=20)

    ax=fig.add_subplot(223)
    im = plt.imshow(10*np.log10(R/R.max()).T, extent=[smin,smax, smin,smax], vmin = min_relative_pow,vmax = 0,origin='lower',cmap='gist_stern_r',interpolation=inter_mode)
    # circle=plt.Circle((0,0),0.3*111.19,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    # circle=plt.Circle((0,0),27,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    plt.title('Radial')
    ax.set_xlabel('East/West Slowness [s/deg]')
    ax.set_ylabel('North/South Slowness [s/deg]')
    cbar = plt.colorbar(im)
    cbar.set_label('relative power (dB)',rotation=270,labelpad=20)

    ax=fig.add_subplot(224)
    im = plt.imshow(10*np.log10(T/T.max()).T, extent=[smin,smax, smin,smax], vmin = min_relative_pow,vmax = 0,origin='lower',cmap='gist_stern_r',interpolation=inter_mode)
    # circle=plt.Circle((0,0),0.3*111.19,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    # circle=plt.Circle((0,0),27,color='w',fill=False,alpha=0.4)
    # plt.gcf().gca().add_artist(circle)
    plt.title('Transverse')
    ax.set_xlabel('East/West Slowness [s/deg]')
    ax.set_ylabel('North/South Slowness [s/deg]')
    cbar = plt.colorbar(im)
    cbar.set_label('relative power (dB)',rotation=270,labelpad=20)
    plt.tight_layout()


    plt.show()



def make_P_Capon(nk,nr,kinc,kmin,rx,ry,icsdm):
    import numpy as np
    import scipy as sp
    from subroutine_CLEAN_3c import rot

    steer = np.zeros((3,3*nr),dtype=complex)
    Y = np.zeros((3,3),dtype=complex)
    polariz = np.zeros((nk,nk,3))
    norm = 1/np.sqrt(nr)
    for i in range(nk):
        kx=-2*np.pi*(kmin+float(i*kinc))
        for j in range(nk):
            ky=-2*np.pi*(kmin+float(j*kinc))
            steer[0,:nr]=np.exp(1j*(kx*(rx[0]-rx)+ky*(ry[0]-ry)))*norm
            steer[1,nr:2*nr] = steer[0,:nr]
            steer[2,2*nr:] = steer[0,:nr]

            theta = np.arctan2(kx,ky)
            
            for k in range(3):
                xres = steer.conj().dot(icsdm[k]).dot(steer.T)

                u,v = np.linalg.eigh(xres)


                uid = u.argsort()

                i0 = uid[0]
                i1 = uid[1]
                #i2 = uid[2]

                z1   = v[0,i0]
                z2   = v[0,i1]
                #z3   = v[0,i2]
                bh11 = v[1,i0] 
                bh12 = v[1,i1] 
                #bh13 = v[1,i2]
                bh21 = v[2,i0] 
                bh22 = v[2,i1] 
                #bh23 = v[2,i2]


                r1,t1 = rot(bh11,bh21,theta)
                r2,t2 = rot(bh12,bh22,theta)
                #r3,t3 = rot(bh13,bh23,theta)

                if k == 0:
                    polariz[i,j,0] = 1 / u[i0].real * np.abs(z1)**2 + 1 / u[i1].real * np.abs(z2)**2 #+ 1 / u[i2].real * np.abs(z3)**2 
                if k == 1:
                    polariz[i,j,1] = 1 / u[i0].real * np.abs(r1)**2 + 1 / u[i1].real * np.abs(r2)**2 #+ 1 / u[i2].real * np.abs(r3)**2
                if k == 2:
                    polariz[i,j,2] = 1 / u[i0].real * np.abs(t1)**2 + 1 / u[i1].real * np.abs(t2)**2 #+ 1 / u[i2].real * np.abs(t3)**2

    return polariz



def make_P_fk(nk,nr,kinc,kmin,rx,ry,icsdm):
    import numpy as np
    import scipy as sp
    from subroutine_CLEAN_3c import rot

    steer = np.zeros((3,3*nr),dtype=complex)
    Y = np.zeros((3,3),dtype=complex)
    polariz = np.zeros((nk,nk,3))
    norm = 1/np.sqrt(nr)
    for i in range(nk):
        kx=-2*np.pi*(kmin+float(i*kinc))
        for j in range(nk):
            ky=-2*np.pi*(kmin+float(j*kinc))
            steer[0,:nr]=np.exp(1j*(kx*(rx[0]-rx)+ky*(ry[0]-ry)))*norm
            steer[1,nr:2*nr] = steer[0,:nr]
            steer[2,2*nr:] = steer[0,:nr]

            theta = np.arctan2(kx,ky)
            
            for k in range(3):
                xres = steer.conj().dot(icsdm[k]).dot(steer.T)

                u,v = np.linalg.eigh(xres)


                uid = u.argsort()[::-1]

                i0 = uid[0]
                i1 = uid[1]
                #i2 = uid[2]

                z1   = v[0,i0]
                z2   = v[0,i1]
                #z3   = v[0,i2]
                bh11 = v[1,i0] 
                bh12 = v[1,i1] 
                #bh13 = v[1,i2]
                bh21 = v[2,i0] 
                bh22 = v[2,i1] 
                #bh23 = v[2,i2]


                r1,t1 = rot(bh11,bh21,theta)
                r2,t2 = rot(bh12,bh22,theta)
                #r3,t3 = rot(bh13,bh23,theta)

                if k == 0:
                    polariz[i,j,0] =  u[i0].real * np.abs(z1)**2 +  u[i1].real * np.abs(z2)**2 #+  u[i2].real * np.abs(z3)**2 
                if k == 1:
                    polariz[i,j,1] =  u[i0].real * np.abs(r1)**2 +  u[i1].real * np.abs(r2)**2 #+  u[i2].real * np.abs(r3)**2
                if k == 2:
                    polariz[i,j,2] =  u[i0].real * np.abs(t1)**2 +  u[i1].real * np.abs(t2)**2 #+  u[i2].real * np.abs(t3)**2

    return polariz



def CLEAN_3C_Capon(nr,max_c,smin,sinc,freq,rx,ry,csdm,control,fk_cln,cln,nk,si):
    import numpy as np
    from subroutine_CLEAN_3c import rot
    steer0 = np.zeros([3,3*nr],dtype=complex)
    steer1 = np.zeros([3,3*nr],dtype=complex)
    steer2 = np.zeros([3,3*nr],dtype=complex)
    icsdm = np.zeros((3,3*nr,3*nr),dtype=complex)


    for c3 in range(3):
        xxs  = np.zeros((3*nr,3*nr),dtype=complex)
        sxopt  = max_c[c3,0]
        syopt  = max_c[c3,1]
        ax     = int((sxopt-smin)/sinc)
        ay     = int((syopt-smin)/sinc)

        steer  = np.exp(-1j*2*np.pi*freq*(sxopt*(rx[0]-rx)+syopt*(ry[0]-ry)))/np.sqrt(nr)
        steer0[0,:nr]     = steer
        steer0[1,nr:2*nr] = steer
        steer0[2,2*nr:]   = steer


        icsdm[c3] = np.linalg.inv(csdm[c3])
        xres = steer0.conj().dot(icsdm[c3]).dot(steer0.T)
        u,v = np.linalg.eigh(xres)
        uid = u.argsort()
        i0 = uid[0]
        i1 = uid[1]



        z1   = v[0,i0]
        z2   = v[0,i1] 

        bh11 = v[1,i0]
        bh12 = v[1,i1]

        bh21 = v[2,i0]
        bh22 = v[2,i1]


        theta = np.arctan2(sxopt,syopt)

        r1,t1 = rot(bh11,bh21,theta)
        r2,t2 = rot(bh12,bh22,theta)


        z1_phase   = np.angle(z1)
        z2_phase   = np.angle(z2)

        bh11_phase = np.angle(bh11)
        bh12_phase = np.angle(bh12)

        bh21_phase = np.angle(bh21)
        bh22_phase = np.angle(bh22)


        baz = np.arctan2(sxopt,syopt)/np.pi*180
        if baz<0:
            baz += 360

        if si == True:
            if cln == 1 and c3 == 0:
                print
                print 'CLEAN iteration |  BAZ    vel | Amplitudes EV0 (Z,R,T) | Amplitudes EV1 (Z,R,T) |   Power of peak EV0 (Z,R,T)   |   Power of peak EV1 (Z,R,T)   |  ZR-Phase EV0 / EV1 |  ZT-Phase EV0 / EV1 |  RT-Phase EV0 / EV1 |'
            print '     %3i        | %5.01f %5.01f |   %.02f   %.02f   %.02f   |   %.02f   %.02f   %.02f   |  %6.02f   %6.02f   %6.02f  |  %6.02f   %6.02f   %6.02f  | %7.02f  / %7.02f  | %7.02f  / %7.02f  | %7.02f  / %7.02f  |' %(
                cln,
                baz,111.19/np.sqrt(sxopt**2 + syopt**2),
                np.abs(z1),np.abs(r1),np.abs(t1),
                np.abs(z2),np.abs(r2),np.abs(t2),
                10*np.log10(np.abs(z1)**2/u[i0].real),10*np.log10(np.abs(r1)**2/u[i0].real),10*np.log10(np.abs(t1)**2/u[i0].real),
                10*np.log10(np.abs(z2)**2/u[i1].real),10*np.log10(np.abs(r2)**2/u[i1].real),10*np.log10(np.abs(t2)**2/u[i1].real),
                np.abs(np.angle(z1,deg=True) - np.angle(r1,deg=True)),np.abs(np.angle(z2,deg=True) - np.angle(r2,deg=True)),
                np.abs(np.angle(z1,deg=True) - np.angle(t1,deg=True)),np.abs(np.angle(z2,deg=True) - np.angle(t2,deg=True)),
                np.abs(np.angle(r1,deg=True) - np.angle(t1,deg=True)),np.abs(np.angle(r2,deg=True) - np.angle(t2,deg=True)))


        steer1[0,:nr]     = np.abs(z1)   * steer * np.exp ( 1j * z1_phase    ) 
        steer1[1,nr:2*nr] = np.abs(bh11) * steer * np.exp ( 1j * bh11_phase  )
        steer1[2,2*nr:]   = np.abs(bh21) * steer * np.exp ( 1j * bh21_phase  )

        steer2[0,:nr]     = np.abs(z2)   * steer * np.exp ( 1j * z2_phase    )
        steer2[1,nr:2*nr] = np.abs(bh12) * steer * np.exp ( 1j * bh12_phase  )
        steer2[2,2*nr:]   = np.abs(bh22) * steer * np.exp ( 1j * bh22_phase  )




        xxs += control/u[i0].real*np.outer(np.sum(steer1,axis=0),np.sum(steer1,axis=0).conj())
        xxs += control/u[i1].real*np.outer(np.sum(steer2,axis=0),np.sum(steer2,axis=0).conj())

        csdm[c3] -= xxs


        if c3 == 0:
            fk_cln[0,ax,ay] += ( np.abs(z1)**2 / u[i0].real + np.abs(z2)**2 / u[i1].real ) * control
        if c3 == 1:
            fk_cln[1,ax,ay] += ( np.abs(r1)**2 / u[i0].real + np.abs(r2)**2 / u[i1].real ) * control
        if c3 == 2:
            fk_cln[2,ax,ay] += ( np.abs(t1)**2 / u[i0].real + np.abs(t2)**2 / u[i1].real ) * control
    return csdm,fk_cln





def CLEAN_3C_fk(nr,max_c,smin,sinc,freq,rx,ry,csdm,control,fk_cln,cln,nk,si):
    import numpy as np
    from subroutine_CLEAN_3c import rot


    steer0 = np.zeros([3,3*nr],dtype=complex)
    steer1 = np.zeros([3,3*nr],dtype=complex)
    steer2 = np.zeros([3,3*nr],dtype=complex)
    icsdm = np.zeros((3,3*nr,3*nr),dtype=complex)


    for c3 in range(3):
        xxs  = np.zeros((3*nr,3*nr),dtype=complex)
        sxopt  = max_c[c3,0]
        syopt  = max_c[c3,1]
        ax     = int((sxopt-smin)/sinc)
        ay     = int((syopt-smin)/sinc)

        steer  = np.exp(-1j*2*np.pi*freq*(sxopt*(rx[0]-rx)+syopt*(ry[0]-ry)))/np.sqrt(nr)
        steer0[0,:nr]     = steer
        steer0[1,nr:2*nr] = steer
        steer0[2,2*nr:]   = steer



        xres = steer0.conj().dot(csdm[c3]).dot(steer0.T)
        u,v = np.linalg.eigh(xres)
        uid = u.argsort()[::-1]
        i0 = uid[0]
        i1 = uid[1]



        z1   = v[0,i0]
        z2   = v[0,i1] 

        bh11 = v[1,i0]
        bh12 = v[1,i1]

        bh21 = v[2,i0]
        bh22 = v[2,i1]


        theta = np.arctan2(sxopt,syopt)

        r1,t1 = rot(bh11,bh21,theta)
        r2,t2 = rot(bh12,bh22,theta)


        z1_phase   = np.angle(z1)
        z2_phase   = np.angle(z2)

        bh11_phase = np.angle(bh11)
        bh12_phase = np.angle(bh12)

        bh21_phase = np.angle(bh21)
        bh22_phase = np.angle(bh22)



        baz = np.arctan2(sxopt,syopt)/np.pi*180
        if baz<0:
            baz += 360

        if si == True:
            if cln == 1 and c3 == 0:
                print
                print 'CLEAN iteration |  BAZ    vel | Amplitudes EV0 (Z,R,T) | Amplitudes EV1 (Z,R,T) |   Power of peak EV0 (Z,R,T)   |   Power of peak EV1 (Z,R,T)   |  ZR-Phase EV0 / EV1 |  ZT-Phase EV0 / EV1 |  RT-Phase EV0 / EV1 |'
            print '     %3i        | %5.01f %5.01f |   %.02f   %.02f   %.02f   |   %.02f   %.02f   %.02f   |  %6.02f   %6.02f   %6.02f  |  %6.02f   %6.02f   %6.02f  | %7.02f  / %7.02f  | %7.02f  / %7.02f  | %7.02f  / %7.02f  |' %(
                cln,
                baz,111.19/np.sqrt(sxopt**2 + syopt**2),
                np.abs(z1),np.abs(r1),np.abs(t1),
                np.abs(z2),np.abs(r2),np.abs(t2),
                10*np.log10(np.abs(z1)**2*u[i0].real),10*np.log10(np.abs(r1)**2*u[i0].real),10*np.log10(np.abs(t1)**2*u[i0].real),
                10*np.log10(np.abs(z2)**2*u[i1].real),10*np.log10(np.abs(r2)**2*u[i1].real),10*np.log10(np.abs(t2)**2*u[i1].real),
                np.abs(np.angle(z1,deg=True) - np.angle(r1,deg=True)),np.abs(np.angle(z2,deg=True) - np.angle(r2,deg=True)),
                np.abs(np.angle(z1,deg=True) - np.angle(t1,deg=True)),np.abs(np.angle(z2,deg=True) - np.angle(t2,deg=True)),
                np.abs(np.angle(r1,deg=True) - np.angle(t1,deg=True)),np.abs(np.angle(r2,deg=True) - np.angle(t2,deg=True)))


        steer1[0,:nr]     = np.abs(z1)   * steer * np.exp ( 1j * z1_phase    ) 
        steer1[1,nr:2*nr] = np.abs(bh11) * steer * np.exp ( 1j * bh11_phase  )
        steer1[2,2*nr:]   = np.abs(bh21) * steer * np.exp ( 1j * bh21_phase  )

        steer2[0,:nr]     = np.abs(z2)   * steer * np.exp ( 1j * z2_phase    )
        steer2[1,nr:2*nr] = np.abs(bh12) * steer * np.exp ( 1j * bh12_phase  )
        steer2[2,2*nr:]   = np.abs(bh22) * steer * np.exp ( 1j * bh22_phase  )




        xxs += control*u[i0].real*np.outer(np.sum(steer1,axis=0),np.sum(steer1,axis=0).conj())
        xxs += control*u[i1].real*np.outer(np.sum(steer2,axis=0),np.sum(steer2,axis=0).conj())

        csdm[c3] -= xxs


        if c3 == 0:
            fk_cln[0,ax,ay] += ( np.abs(z1)**2 * u[i0].real + np.abs(z2)**2 * u[i1].real ) * control
        if c3 == 1:
            fk_cln[1,ax,ay] += ( np.abs(r1)**2 * u[i0].real + np.abs(r2)**2 * u[i1].real ) * control
        if c3 == 2:
            fk_cln[2,ax,ay] += ( np.abs(t1)**2 * u[i0].real + np.abs(t2)**2 * u[i1].real ) * control
    return csdm,fk_cln






def make_csdm(nwin,nr,xt,nsamp,find,fave):
    import numpy as np
    csdm = np.zeros((3,3*nr,3*nr),dtype=complex)
    ffts = np.zeros((nr*3,nsamp/2+1),dtype = complex)
    for i in range(nwin):
        for j in range(3):
            for k in range(nr):
                ffts[k+nr*j] = np.fft.rfft(xt[j,k,i],nsamp)
        for m in range(find-fave,find+fave+1):
            csdm[0] += np.outer(ffts[:,m],ffts[:,m].T.conj())
    print 'Normalization is adapted to Hann window'
    csdm[0] /= nwin * ( 2 * fave +1 ) * nsamp * nr * 3/8.  #3/8 is the fraction to take into account the Hann window power recudction.
    csdm[1] = np.copy(csdm[0])
    csdm[2] = np.copy(csdm[0])
    return csdm



def make_subwindows_PSAR(nr,nwin,pdic,st,st0,st1,nsamp):
    import numpy as np
    from subroutine_CLEAN_3c import tap
    xt = np.zeros((3,nr,nwin,nsamp))
    for i in range(nr):
        for j in range(nwin):
            theta = pdic[st0[i].stats.station]/180.*np.pi
            xt[0,i,j] = st[i][j*nsamp/2:(j+2)*nsamp/2]
            xt[1,i,j] = st0[i][j*nsamp/2:(j+2)*nsamp/2]*np.cos(theta) - st1[i][j*nsamp/2:(j+2)*nsamp/2]*np.sin(theta)
            xt[2,i,j] = st0[i][j*nsamp/2:(j+2)*nsamp/2]*np.sin(theta) + st1[i][j*nsamp/2:(j+2)*nsamp/2]*np.cos(theta)
            xt[0,i,j] -= np.mean(xt[0,i,j])
            xt[1,i,j] -= np.mean(xt[1,i,j])
            xt[2,i,j] -= np.mean(xt[2,i,j])
            xt[0,i,j] = tap(xt[0,i,j],nsamp)
            xt[1,i,j] = tap(xt[1,i,j],nsamp)
            xt[2,i,j] = tap(xt[2,i,j],nsamp)
    return xt


def make_subwindows(nr,nwin,st,st0,st1,nsamp):
    import numpy as np
    from subroutine_CLEAN_3c import tap
    xt = np.zeros((3,nr,nwin,nsamp))
    for i in range(nr):
        for j in range(nwin):
            xt[0,i,j] = st[i][j*nsamp/2:(j+2)*nsamp/2]
            xt[1,i,j] = st0[i][j*nsamp/2:(j+2)*nsamp/2]
            xt[2,i,j] = st1[i][j*nsamp/2:(j+2)*nsamp/2]
            xt[0,i,j] -= np.mean(xt[0,i,j])
            xt[1,i,j] -= np.mean(xt[1,i,j])
            xt[2,i,j] -= np.mean(xt[2,i,j])
            xt[0,i,j] = tap(xt[0,i,j],nsamp)
            xt[1,i,j] = tap(xt[1,i,j],nsamp)
            xt[2,i,j] = tap(xt[2,i,j],nsamp)
    return xt


def PSAR_dict():
    pdic = dict()
    pdic['PSA00'] =  289.8
    pdic['PSAA1'] =  350.8
    pdic['PSAA2'] =  276.4
    pdic['PSAA3'] =  174.5
    pdic['PSAB1'] =  264.5
    pdic['PSAB2'] =  45.8
    pdic['PSAB3'] =  110.0
    pdic['PSAC1'] =  1.5
    pdic['PSAC2'] =  250.3
    pdic['PSAC3'] =  129.5
    pdic['PSAD1'] =  297.9
    pdic['PSAD2'] =  248.0
    pdic['PSAD3'] =  161.0
    return pdic




def metric_mseed(st,d,nr):
    import numpy as np
    import scipy as sp
    from subroutine_CLEAN_3c import grt
    '''
    function takes data matrix and returns interstation distances rx and ry (vectors) in [deg].
    '''
    rx_0,ry_0 = d[st[0].stats.station]
    rx = np.zeros(nr)
    ry = np.zeros(nr)
    for i in range(nr):
        rx_i,ry_i = d[st[i].stats.station]
        decl,dist,az,baz = grt(float(rx_0),float(ry_0),float(rx_i),float(ry_i))
        #decl,dist,az,baz = ut.grt(st[0].stats.sac.stla,st[0].stats.sac.stlo,st[i].stats.sac.stla,st[i].stats.sac.stlo)
        rx[i] = decl*sp.cos(0.017453*(90.0-az))
        ry[i] = decl*sp.sin(0.017453*(90.0-az))
    return rx,ry



def get_metadata(meta_f):
    d = dict()
    with open(meta_f) as f:
        for line in f:
            x = line.split('|')
            d[x[1]] = x[4],x[5]
    return d

def get_metadata_NORSAR(meta_f):
    d = dict()
    with open(meta_f) as f:
        for line in f:
            x = line.split(' ')
            x = filter(None, x) 
            d[x[0]] = x[3],x[4]
    return d




def gkern2(kernlen=21, nsig=3):
    import numpy as np
    import scipy.ndimage.filters as filters

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    x = filters.gaussian_filter(inp, nsig)

    return x/x.max()

def rot(x,y,theta):
    import numpy as np
    a = ( x*np.cos(theta) + y*np.sin(theta))
    b = (-x*np.sin(theta) + y*np.cos(theta))
    return a,b



def tap(ff,nsamp):
    import numpy as np

    x = np.hanning(nsamp)
    ff = ff*x
    return ff

def grt(r1,r2,s1,s2):
    import numpy as np
    import math as m
    slat=s1*np.pi/180.
    slon=s2*np.pi/180.
    elat=r1*np.pi/180.
    elon=r2*np.pi/180.
    
    slat=m.atan(.996647*m.tan(slat))
    elat=m.atan(.996647*m.tan(elat))


    slat=np.pi/2.0-slat
    elat=np.pi/2.0-elat


    if(slon<0.0):
        slon+=2.0*np.pi
    if(elon<0.0):
        elon+=2.0*np.pi



    a=m.sin(elat)*m.cos(elon)
    b=m.sin(elat)*m.sin(elon)
    c=m.cos(elat)
    a1=m.sin(slat)*m.cos(slon)
    b1=m.sin(slat)*m.sin(slon)
    c1=m.cos(slat)

    cd=a*a1+b*b1+c*c1

    if(cd>1.0):
        cd=1.0
    if(cd<-1.0):
        cd=-1.0
    decl=m.acos(cd)*180.0/m.pi
    dist=decl*np.pi*6371.0/180.0

    tmp1=m.cos(elon)*m.cos(slon)+m.sin(elon)*m.sin(slon)
    tmp2a=1.0-cd*cd

    if tmp2a<=0.0:
        tmp2=0.0
        tmp3=1.0
    else:
        tmp2=m.sqrt(tmp2a)
        tmp3=(m.sin(elat)*m.cos(slat)-m.cos(elat)*m.sin(slat)*tmp1)/tmp2

    if(tmp3>1.0):
        tmp3=1.0
    if(tmp3<-1.0):
        tmp3=-1.0
    z=m.acos(tmp3)

    if((m.sin(slon)*m.cos(elon)-m.cos(slon)*m.sin(elon))<0.0):
        z=2.0*m.pi-z

    az=180.0*z/m.pi

    tmp1=m.cos(slon)*m.cos(elon)+m.sin(slon)*m.sin(elon)
    tmp2a=1.0-cd*cd
    if(tmp2a<=0.0):
        tmp2=0.0
        tmp3=1.0
    else: 
        tmp2=m.sqrt(tmp2a)
        tmp3=(m.sin(slat)*m.cos(elat)-m.cos(slat)*m.sin(elat)*tmp1)/tmp2

    

    if(tmp3>1.0):
        tmp3=1.0
    if(tmp3<-1.0):
        tmp3=-1.0
        
    bz=m.acos(tmp3)

    if((m.sin(elon)*m.cos(slon)-m.cos(elon)*m.sin(slon))<0.0):
        bz=2.0*m.pi-bz
        

    baz=180.0*bz/m.pi
    
    return decl,dist,az,baz

def get_path_mseed_3C(station,d,i,folder,year):
    tmp1 = '%s%s.%s.%03d.%02d.00.%s.mseed' % (folder,station,year,d+1,i,'BHZ')
    tmp2 = '%s%s.%s.%03d.%02d.00.%s.mseed' % (folder,station,year,d+1,i,'BH1')
    tmp3 = '%s%s.%s.%03d.%02d.00.%s.mseed' % (folder,station,year,d+1,i,'BH2')
    return tmp1,tmp2,tmp3
    


def f_output_clean(fk,d,ii,rmvd):
    import numpy as np
    import scipy as sp
    import scipy.ndimage.filters as filters
    
    r = []


    maxxi = (np.where(fk==filters.maximum_filter(fk, 5)))
    this=np.empty([2,len(maxxi[0])])
    rg_l = []

    for i in range(len(maxxi[0])):
     this[0][i]=(maxxi[0][i]-80)*0.5
     this[1][i]=(maxxi[1][i]-80)*0.5
     if (10*np.log10(fk[maxxi[0][i],maxxi[1][i]]/fk.max()) > rmvd):
         baz=np.math.atan2(this[0][i],this[1][i])*180.0/3.1415926
         if(baz<0.0):
             baz+=360.0
         xvel = 111.19/sp.sqrt(this[0][i]**2+this[1][i]**2)
         xamp = fk[maxxi[0][i],maxxi[1][i]]

         rg_l.append([xamp,xvel,baz])


    rg_l.sort(reverse=True)

    return d,ii,rg_l


def write_result(res,d,i,Z,R,T,freq):
    import numpy as np
    fh  = open('%sday_%s_hour_%s_freq_%.03f.txt'%(res,d+1,i,freq),'w')
    day,h,arrivals = f_output_clean(Z,d+1,i,-10)
    fh.write ('%i %i %s\n'%(day,h,'Z'))
    for ia in arrivals:
        if(ia !=[]):
            fh.write ('%.02f %.02f %.02f %.02f\n'%(10*np.log10(ia[0]),10*np.log10(ia[0]/(Z.max())),ia[1],ia[2]))
    day,h,arrivals = f_output_clean(R,d+1,i,-10)
    fh.write ('%i %i %s\n'%(day,h,'R'))
    for ia in arrivals:
        if(ia !=[]):
            fh.write ('%.02f %.02f %.02f %.02f\n'%(10*np.log10(ia[0]),10*np.log10(ia[0]/(R.max())),ia[1],ia[2]))
    day,h,arrivals = f_output_clean(T,d+1,i,-10)
    fh.write ('%i %i %s\n'%(day,h,'T'))
    for ia in arrivals:
        if(ia !=[]):
            fh.write ('%.02f %.02f %.02f %.02f\n'%(10*np.log10(ia[0]),10*np.log10(ia[0]/(T.max())),ia[1],ia[2]))
    fh.close()


def sta_pop(i,st,st0,st1):
    idd = 0
    for j in st:
        if str(j.stats.station) == i:
            st.pop(idd)
        idd += 1

    idd = 0
    for j in st0:
        if str(j.stats.station) == i:
            st0.pop(idd)
        idd += 1

    idd = 0
    for j in st1:
        if str(j.stats.station) == i:
            st1.pop(idd)
        idd += 1
    return st,st0,st1


def equalize(st,st0,st1):
    from collections import defaultdict
    dsta = dict()
    dsta = defaultdict(lambda: 0, dsta)
    for i in st:
        dsta[str(i.stats.station)] += 1
    for i in st0:
        dsta[str(i.stats.station)] += 1
    for i in st1:
        dsta[str(i.stats.station)] += 1

    for i in dsta.keys():
        if dsta[i] != 3:
            #print i 
            sta_pop(i,st,st0,st1)
    return st,st0,st1