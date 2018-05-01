import numpy as np
cimport numpy as np
cimport cython


DTYPE1 = np.float64
ctypedef np.float64_t DTYPE1_t

DTYPE2 = np.complex128
ctypedef np.complex128_t DTYPE2_t


ctypedef double complex complex128_t
ctypedef double double_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef dAdzmm_ron_s1_cython(const complex128_t[:,::1] u0,const complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, const complex128_t[:,::1] Q,
                    const double_t tsh, double_t dt, const complex128_t[:,::1] hf,
                    const double_t[:,::1] w_tiled, const complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef int j
    cdef double_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype=DTYPE1)
    cdef complex128_t[:,::1] M4 = np.empty([shapeM2,shape2], dtype=DTYPE2)
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype=DTYPE2)
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = (u0[M2[0,i],j]*u0_conj[M2[1,i],j]).real
    
    M4 = np.fft.fft(M3)
    for i in range(shapeM2):
        for j in range(shape2):
            M4[i,j] = M4[i,j] * hf[i,j]
    
    M4 = dt*np.fft.fftshift(np.fft.ifft(M4))

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j] + \
                               0.54*Q[0,i]*M4[M1[4,i],j])
    cdef complex128_t[:,::1] M5 = np.fft.fft(N)
    for i in range(shape1):
        for j in range(shape2):
            M5[i,j] = w_tiled[i,j] * M5[i,j]
    M5 = tsh * np.fft.ifft(M5)
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * ( N[i,j]  + M5[i,j])
    return N



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef dAdzmm_ron_s0_cython(const complex128_t[:,::1] u0,const complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, const complex128_t[:,::1] Q,
                    const double_t tsh, double_t dt, const complex128_t[:,::1] hf,
                    const double_t[:,::1] w_tiled, const complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef int j
    cdef double_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype=DTYPE1)
    cdef complex128_t[:,::1] M4 = np.empty([shapeM2,shape2], dtype=DTYPE2)
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype=DTYPE2)
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = (u0[M2[0,i],j]*u0_conj[M2[1,i],j]).real
    
    M4 = np.fft.fft(M3)
    for i in range(shapeM2):
        for j in range(shape2):
            M4[i,j] = M4[i,j] * hf[i,j]
    
    M4 = dt*np.fft.fftshift(np.fft.ifft(M4))

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j] + \
                               0.54*Q[0,i]*M4[M1[4,i],j])
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * N[i,j]
    return N

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef dAdzmm_roff_s0_cython(const complex128_t[:,::1] u0,const complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, const complex128_t[:,::1] Q,
                    const double_t tsh, double_t dt, const complex128_t[:,::1] hf,
                    const double_t[:,::1] w_tiled, const complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef int j
    cdef double_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype=DTYPE1)
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype=DTYPE2)
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = (u0[M2[0,i],j]*u0_conj[M2[1,i],j]).real
    

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j])
                              
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * N[i,j]
    return N


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef dAdzmm_roff_s1_cython(const complex128_t[:,::1] u0,const complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, const complex128_t[:,::1] Q,
                    const double_t tsh, double_t dt, const complex128_t[:,::1] hf,
                    const double_t[:,::1] w_tiled, const complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef int j
    cdef double_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype=DTYPE1)
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype=DTYPE2)
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = (u0[M2[0,i],j]*u0_conj[M2[1,i],j]).real
    

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j])
                              
    cdef complex128_t[:,::1] M5 = np.fft.fft(N)
    for i in range(shape1):
        for j in range(shape2):
            M5[i,j] = w_tiled[i,j] * M5[i,j]
    M5 = tsh * np.fft.ifft(M5)
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * ( N[i,j]  + M5[i,j])
    return N
