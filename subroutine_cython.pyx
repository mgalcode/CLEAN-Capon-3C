import numpy as np
import scipy as sp
from cython import boundscheck, wraparound





cdef extern from "math.h":
    float cosf(float x)
    float sinf(float x)
    double cos(double x)
    double sin(double x)
    float sqrtf(float x)
    float atan2f(float x,float y)
cdef extern from "complex.h":
    double cabs(double complex z)
    double creal(double complex z)
    double cimag(double complex z)
    double complex cexp(double complex z)
    double complex conj(double complex z);





def rot_cython(complex x, complex y, float theta):
    cdef complex a,b
    a = ( x*cosf(theta) + y*sinf(theta))
    b = (-x*sinf(theta) + y*cosf(theta))
    return a,b

cdef rot_c_r(complex x, complex y, float theta):
    return ( x*cosf(theta) + y*sinf(theta))

cdef rot_c_t(complex x, complex y, float theta):
    return (-x*sinf(theta) + y*cosf(theta))




def get_max_cython(polariz,float smin,float sinc,int cln):
    cdef int i,j,k
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


def CLEAN_3C_Capon_cython(int nr,max_c,smin,sinc,freq,rx,ry,csdm,control,fk_cln,cln,int nk,si):
    from subroutine_cython import rot_cython as rot

    
    cdef int c3

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
                cabs(z1),cabs(r1),cabs(t1),
                cabs(z2),cabs(r2),cabs(t2),
                10*np.log10(cabs(z1)**2/u[i0].real),10*np.log10(cabs(r1)**2/u[i0].real),10*np.log10(cabs(t1)**2/u[i0].real),
                10*np.log10(cabs(z2)**2/u[i1].real),10*np.log10(cabs(r2)**2/u[i1].real),10*np.log10(cabs(t2)**2/u[i1].real),
                cabs(np.angle(z1,deg=True) - np.angle(r1,deg=True)),cabs(np.angle(z2,deg=True) - np.angle(r2,deg=True)),
                cabs(np.angle(z1,deg=True) - np.angle(t1,deg=True)),cabs(np.angle(z2,deg=True) - np.angle(t2,deg=True)),
                cabs(np.angle(r1,deg=True) - np.angle(t1,deg=True)),cabs(np.angle(r2,deg=True) - np.angle(t2,deg=True)))


        steer1[0,:nr]     = cabs(z1)   * steer * np.exp ( 1j * z1_phase    ) 
        steer1[1,nr:2*nr] = cabs(bh11) * steer * np.exp ( 1j * bh11_phase  )
        steer1[2,2*nr:]   = cabs(bh21) * steer * np.exp ( 1j * bh21_phase  )

        steer2[0,:nr]     = cabs(z2)   * steer * np.exp ( 1j * z2_phase    )
        steer2[1,nr:2*nr] = cabs(bh12) * steer * np.exp ( 1j * bh12_phase  )
        steer2[2,2*nr:]   = cabs(bh22) * steer * np.exp ( 1j * bh22_phase  )




        xxs += control/u[i0].real*np.outer(np.sum(steer1,axis=0),np.sum(steer1,axis=0).conj())
        xxs += control/u[i1].real*np.outer(np.sum(steer2,axis=0),np.sum(steer2,axis=0).conj())

        csdm[c3] -= xxs


        if c3 == 0:
            fk_cln[0,ax,ay] += ( cabs(z1)**2 / u[i0].real + cabs(z2)**2 / u[i1].real ) * control
        if c3 == 1:
            fk_cln[1,ax,ay] += ( cabs(r1)**2 / u[i0].real + cabs(r2)**2 / u[i1].real ) * control
        if c3 == 2:
            fk_cln[2,ax,ay] += ( cabs(t1)**2 / u[i0].real + cabs(t2)**2 / u[i1].real ) * control
    return csdm,fk_cln

def CLEAN_3C_fk_cython(int nr,max_c,smin,sinc,freq,rx,ry,csdm,control,fk_cln,cln,int nk,si):
    from subroutine_cython import rot_cython as rot

    
    cdef int c3

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
                cabs(z1),cabs(r1),cabs(t1),
                cabs(z2),cabs(r2),cabs(t2),
                10*np.log10(cabs(z1)**2*u[i0].real),10*np.log10(cabs(r1)**2*u[i0].real),10*np.log10(cabs(t1)**2*u[i0].real),
                10*np.log10(cabs(z2)**2*u[i1].real),10*np.log10(cabs(r2)**2*u[i1].real),10*np.log10(cabs(t2)**2*u[i1].real),
                cabs(np.angle(z1,deg=True) - np.angle(r1,deg=True)),cabs(np.angle(z2,deg=True) - np.angle(r2,deg=True)),
                cabs(np.angle(z1,deg=True) - np.angle(t1,deg=True)),cabs(np.angle(z2,deg=True) - np.angle(t2,deg=True)),
                cabs(np.angle(r1,deg=True) - np.angle(t1,deg=True)),cabs(np.angle(r2,deg=True) - np.angle(t2,deg=True)))


        steer1[0,:nr]     = cabs(z1)   * steer * np.exp ( 1j * z1_phase    ) 
        steer1[1,nr:2*nr] = cabs(bh11) * steer * np.exp ( 1j * bh11_phase  )
        steer1[2,2*nr:]   = cabs(bh21) * steer * np.exp ( 1j * bh21_phase  )

        steer2[0,:nr]     = cabs(z2)   * steer * np.exp ( 1j * z2_phase    )
        steer2[1,nr:2*nr] = cabs(bh12) * steer * np.exp ( 1j * bh12_phase  )
        steer2[2,2*nr:]   = cabs(bh22) * steer * np.exp ( 1j * bh22_phase  )




        xxs += control*u[i0].real*np.outer(np.sum(steer1,axis=0),np.sum(steer1,axis=0).conj())
        xxs += control*u[i1].real*np.outer(np.sum(steer2,axis=0),np.sum(steer2,axis=0).conj())

        csdm[c3] -= xxs


        if c3 == 0:
            fk_cln[0,ax,ay] += ( cabs(z1)**2 * u[i0].real + cabs(z2)**2 * u[i1].real ) * control
        if c3 == 1:
            fk_cln[1,ax,ay] += ( cabs(r1)**2 * u[i0].real + cabs(r2)**2 * u[i1].real ) * control
        if c3 == 2:
            fk_cln[2,ax,ay] += ( cabs(t1)**2 * u[i0].real + cabs(t2)**2 * u[i1].real ) * control
    return csdm,fk_cln

def make_csdm_cython(int nwin,int nr,xt,int nsamp,int find,int fave):
    cdef int i,j,k,m
    csdm = np.zeros((3,3*nr,3*nr),dtype=complex)
    ffts = np.zeros((nr*3,nsamp/2+1),dtype = complex)
    for i in range(nwin):
        for j in range(3):
            for k in range(nr):
                ffts[k+nr*j] = np.fft.rfft(xt[j,k,i],nsamp)
        for m in range(find-fave,find+fave+1):
            csdm[0] += np.outer(ffts[:,m],ffts[:,m].T.conj())
            
    print 'Normalization is adapted to Hann window'
    csdm[0] /= nwin * ( 2 * fave +1 ) * nsamp * nr * 3/8.
    csdm[1] = np.copy(csdm[0])
    csdm[2] = np.copy(csdm[0])
    return csdm




@boundscheck(False)
@wraparound(False)
def make_P_Capon_cython_v3(int nk,int nr,double kinc,double kmin,double[:] rx,double[:] ry,double complex[:,:,:] icsdm):
    import numpy as np
    import scipy as sp
    from subroutine_cython import rot_cython as rot


    cdef int i,j,k,i0,i1,m,n,ix1,ix2,ix3,ix4
    cdef complex z1,z2,bh11,bh12,bh21,bh22,r1,t1,r2,t2
    cdef double complex[:,:] Y = np.zeros((3,3),dtype=complex)
    #cdef double complex[:,:] v = np.empty((3,3),dtype=complex) # this is slower!
    cdef double complex carg,carg_c
    cdef double[:,:,:] polariz = np.zeros((nk,nk,3))
    cdef float norm = 1/float(nr),theta,u0,u1
    cdef double kx,ky,arg

    kmin *=2*3.1415926
    kinc *=2*3.1415926
    for i in range(nk):
        kx=-(kmin+(i*kinc))
        for j in range(nk):
            ky=-(kmin+(j*kinc))
            theta = atan2f(kx,ky)
            
            for k in range(3):

                for m in range(nr):
                    ix1 = m+nr
                    ix2 = m+2*nr
                    Y[0,0] += creal(icsdm[k,m,m])
                    Y[1,1] += creal(icsdm[k,ix1,ix1])
                    Y[2,2] += creal(icsdm[k,ix2,ix2])

                    Y[0,1] += icsdm[k,m,ix1]
                    Y[0,2] += icsdm[k,m,ix2]
                    Y[1,2] += icsdm[k,ix1,ix2]

                    for n in range(m+1,nr):
                        ix3    = n+nr
                        ix4    = n+2*nr
                        arg    = kx*(rx[m]-rx[n])+ky*(ry[m]-ry[n])
                        carg   = cexp(1j*arg) 
                        carg_c = conj(carg)
                        Y[0,0]+= 2.0*(creal(icsdm[k,m,n])*cos(arg)-cimag(icsdm[k,m,n])*sin(arg))
                        Y[1,1]+= 2.0*(creal(icsdm[k,ix1,ix3])*cos(arg)-cimag(icsdm[k,ix1,ix3])*sin(arg))
                        Y[2,2]+= 2.0*(creal(icsdm[k,ix2,ix4])*cos(arg)-cimag(icsdm[k,ix2,ix4])*sin(arg))

                        Y[0,1]+= icsdm[k,m,ix3]*carg + carg_c*icsdm[k,n,ix1]
                        Y[0,2]+= icsdm[k,m,ix4]*carg + carg_c*icsdm[k,n,ix2]
                        Y[1,2]+= icsdm[k,ix1,ix4]*carg + carg_c*icsdm[k,ix3,ix2]



                Y[0,0] *= norm
                Y[0,1] *= norm
                Y[0,2] *= norm
                Y[1,1] *= norm
                Y[1,2] *= norm
                Y[2,2] *= norm



                u,v = np.linalg.eigh(Y,UPLO='U')

                uid = u.argsort()

                i0 = uid[0]
                i1 = uid[1]


                z1   = v[0,i0]
                z2   = v[0,i1]

                bh11 = v[1,i0] 
                bh12 = v[1,i1] 

                bh21 = v[2,i0] 
                bh22 = v[2,i1] 



                r1 = rot_c_r(bh11,bh21,theta)
                t1 = rot_c_t(bh11,bh21,theta)

                r2 = rot_c_r(bh12,bh22,theta)
                t2 = rot_c_t(bh12,bh22,theta)
                

                u0 = u[i0].real
                u1 = u[i1].real

                if k == 0:
                    polariz[i,j,0] = 1 / u0 * cabs(z1)**2 + 1 / u1 * cabs(z2)**2 #+ 1 / u[i2].real * cabs(z3)**2 
                if k == 1:
                    polariz[i,j,1] = 1 / u0 * cabs(r1)**2 + 1 / u1 * cabs(r2)**2 #+ 1 / u[i2].real * cabs(r3)**2
                if k == 2:
                    polariz[i,j,2] = 1 / u0 * cabs(t1)**2 + 1 / u1 * cabs(t2)**2 #+ 1 / u[i2].real * cabs(t3)**2
                for m in range(3):
                    for n in range(3):
                        Y[m,n] = 0 
    return np.array(polariz)


@boundscheck(False)
@wraparound(False)
def make_P_fk_cython_v3(int nk,int nr,double kinc,double kmin,double[:] rx,double[:] ry,double complex[:,:,:] icsdm):
    import numpy as np
    import scipy as sp
    from subroutine_cython import rot_cython as rot


    cdef int i,j,k,i0,i1,m,n,ix1,ix2,ix3,ix4
    cdef complex z1,z2,bh11,bh12,bh21,bh22,r1,t1,r2,t2
    cdef double complex[:,:] Y = np.zeros((3,3),dtype=complex)
    #cdef double complex[:,:] v = np.empty((3,3),dtype=complex) # this is slower!
    cdef double complex carg,carg_c
    cdef double[:,:,:] polariz = np.zeros((nk,nk,3))
    cdef float norm = 1/float(nr),theta,u0,u1
    cdef double kx,ky,arg

    kmin *=2*3.1415926
    kinc *=2*3.1415926
    for i in range(nk):
        kx=-(kmin+(i*kinc))
        for j in range(nk):
            ky=-(kmin+(j*kinc))
            theta = atan2f(kx,ky)
            
            for k in range(3):

                for m in range(nr):
                    ix1 = m+nr
                    ix2 = m+2*nr
                    Y[0,0] += creal(icsdm[k,m,m])
                    Y[1,1] += creal(icsdm[k,ix1,ix1])
                    Y[2,2] += creal(icsdm[k,ix2,ix2])

                    Y[0,1] += icsdm[k,m,ix1]
                    Y[0,2] += icsdm[k,m,ix2]
                    Y[1,2] += icsdm[k,ix1,ix2]

                    for n in range(m+1,nr):
                        ix3    = n+nr
                        ix4    = n+2*nr
                        arg    = kx*(rx[m]-rx[n])+ky*(ry[m]-ry[n])
                        carg   = cexp(1j*arg) 
                        carg_c = conj(carg)
                        Y[0,0]+= 2.0*(creal(icsdm[k,m,n])*cos(arg)-cimag(icsdm[k,m,n])*sin(arg))
                        Y[1,1]+= 2.0*(creal(icsdm[k,ix1,ix3])*cos(arg)-cimag(icsdm[k,ix1,ix3])*sin(arg))
                        Y[2,2]+= 2.0*(creal(icsdm[k,ix2,ix4])*cos(arg)-cimag(icsdm[k,ix2,ix4])*sin(arg))

                        Y[0,1]+= icsdm[k,m,ix3]*carg + carg_c*icsdm[k,n,ix1]
                        Y[0,2]+= icsdm[k,m,ix4]*carg + carg_c*icsdm[k,n,ix2]
                        Y[1,2]+= icsdm[k,ix1,ix4]*carg + carg_c*icsdm[k,ix3,ix2]



                Y[0,0] *= norm
                Y[0,1] *= norm
                Y[0,2] *= norm
                Y[1,1] *= norm
                Y[1,2] *= norm
                Y[2,2] *= norm



                u,v = np.linalg.eigh(Y,UPLO='U')

                uid = u.argsort()[::-1]

                i0 = uid[0]
                i1 = uid[1]


                z1   = v[0,i0]
                z2   = v[0,i1]

                bh11 = v[1,i0] 
                bh12 = v[1,i1] 

                bh21 = v[2,i0] 
                bh22 = v[2,i1] 



                r1 = rot_c_r(bh11,bh21,theta)
                t1 = rot_c_t(bh11,bh21,theta)

                r2 = rot_c_r(bh12,bh22,theta)
                t2 = rot_c_t(bh12,bh22,theta)
                

                u0 = u[i0].real
                u1 = u[i1].real

                if k == 0:
                    polariz[i,j,0] =  u0 * cabs(z1)**2 +  u1 * cabs(z2)**2 #+  u[i2].real * cabs(z3)**2 
                if k == 1:
                    polariz[i,j,1] =  u0 * cabs(r1)**2 +  u1 * cabs(r2)**2 #+  u[i2].real * cabs(r3)**2
                if k == 2:
                    polariz[i,j,2] =  u0 * cabs(t1)**2 +  u1 * cabs(t2)**2 #+  u[i2].real * cabs(t3)**2
                for m in range(3):
                    for n in range(3):
                        Y[m,n] = 0 
    return np.array(polariz)