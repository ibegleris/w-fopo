#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True
cimport numpy as np
import numpy as np



ctypedef double complex complex128_t
#ctypedef double double


cpdef complex128_t[:,::1] dAdzmm_ron_s1_cython(complex128_t[:,::1] u0 ,complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, complex128_t[:,::1] Q,
                    double tsh, double dt, complex128_t[:,::1] hf,
                    double[:,::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]

    cdef long i, j

    cdef complex128_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype='complex_')
 
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = u0[M2[0,i],j]*u0_conj[M2[1,i],j]

    cdef complex128_t[:,::1] M4 = np.fft.fft(M3)
    

    for i in range(shapeM2):
        for j in range(shape2):
            M4[i,j] = M4[i,j] * hf[i,j]
    
    M4 = np.fft.fftshift(np.fft.ifft(M4), axes= -1)

    cdef complex128_t[::1] Q_comp = np.empty(shapeM1, dtype = 'complex_')
    dt = dt*0.54
    

    
    for i in range(shapeM1):
        Q_comp[i] = 1.64*Q[0,i] + 0.82*Q[1,i]

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(Q_comp[i] * M3[M1[4,i],j] + \
                               dt*Q[0,i]*M4[M1[4,i],j])
    

    cdef complex128_t[:,::1] M5 = np.fft.fft(N)
        
    
    for i in range(shape1):
        for j in range(shape2):
            M5[i,j] = w_tiled[i,j] * M5[i,j]
    
    M5 =  np.fft.ifft(M5)
    
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * ( N[i,j]  + tsh * M5[i,j])
    return N



cpdef complex128_t[:,::1] dAdzmm_ron_s0_cython(complex128_t[:,::1] u0,const complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, complex128_t[:,::1] Q,
                    double tsh, double dt, complex128_t[:,::1] hf,
                    double[:,::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i, j
    cdef complex128_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] M4 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype='complex_')
    
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = u0[M2[0,i],j]*u0_conj[M2[1,i],j]
    
    M4 = np.fft.fft(M3)
    for i in range(shapeM2):
        for j in range(shape2):
            M4[i,j] = M4[i,j] * hf[i,j]
    
    M4 = np.fft.fftshift(np.fft.ifft(M4), axes = -1)

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j] + \
                               dt * 0.54*Q[0,i]*M4[M1[4,i],j])
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * N[i,j]
    return np.asarray(N)

cpdef complex128_t[:,::1] dAdzmm_roff_s0_cython(complex128_t[:,::1] u0,complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, complex128_t[:,::1] Q,
                    double tsh, double dt, complex128_t[:,::1] hf,
                    double[:,::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i, j
    cdef complex128_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] M4 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype='complex_')
    
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = u0[M2[0,i],j]*u0_conj[M2[1,i],j]
    

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j])
                              
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * N[i,j]
    return np.asarray(N)


cpdef complex128_t[:,::1] dAdzmm_roff_s1_cython(const complex128_t[:,::1] u0,const complex128_t[:,::1] u0_conj ,
                    np.ndarray[long, ndim = 2] M1, np.ndarray[long, ndim = 2] M2, const complex128_t[:,::1] Q,
                    const double tsh, double dt, const complex128_t[:,::1] hf,
                    const double[:,::1] w_tiled, const complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i, j
    
    cdef complex128_t[:,::1] M3 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] M4 = np.empty([shapeM2,shape2], dtype='complex_')
    cdef complex128_t[:,::1] N = np.zeros([shape1,shape2], dtype='complex_')
    

    for i in range(shapeM2):
        for j in range(shape2):
            M3[i,j] = u0[M2[0,i],j]*u0_conj[M2[1,i],j]
    

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0,i],j] = N[M1[0,i],j] + u0[M1[1,i],j]\
                            *(0.82*(2*Q[0,i] + Q[1,i]) \
                                *M3[M1[4,i],j])
                              
    M5 = np.fft.fft(N)
    
    for i in range(shape1):
        for j in range(shape2):
            M5[i,j] = w_tiled[i,j] * M5[i,j]
    
    M5 =  np.fft.ifft(M5)
    
    for i in range(shape1):
        for j in range(shape2):
            N[i,j] = gam_no_aeff * ( N[i,j]  + tsh * M5[i,j])
    return np.asarray(N)
