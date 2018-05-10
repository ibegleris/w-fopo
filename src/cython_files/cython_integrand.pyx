#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free





ctypedef double complex complex128_t
#ctypedef double double


cpdef complex128_t[:,::1] dAdzmm_ron_s1_cython(complex128_t[:,::1] u0 ,complex128_t[:,::1] u0_conj ,
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

    cdef complex128_t* M3ptr = &M3[0,0]
    cdef complex128_t* u0ptr = &u0[0,0]
    cdef complex128_t* u0_cptr =  &u0_conj[0,0]

    cdef long* M1ptr = &M1[0,0]
    cdef long* M2ptr = &M2[0,0]
 
    for i in range(shapeM2):
        for j in range(shape2):
            M3ptr[i * shape2 + j] = u0ptr[M2ptr[i] * shape2 + j] *\
                                    u0_cptr[M2ptr[shapeM2 + i] * shape2 + j]
    
    M4 = np.fft.fft(M3)
    cdef complex128_t* M4ptr = &M4[0,0]
    cdef complex128_t* hfptr = &hf[0,0]
    for i in range(shapeM2):
        for j in range(shape2):
            M4ptr[i * shape2 + j] = M4ptr[i * shape2 + j] * hfptr[i * shape2 + j]
    
    M4 = np.fft.fftshift(np.fft.ifft(M4))

    M4ptr = &M4[0,0]
    cdef complex128_t* Nptr = &N[0,0]
    cdef complex128_t* Qptr = &Q[0,0]

    for i in range(shapeM1):
        for j in range(shape2):            
            Nptr[M1ptr[i] * shape2 + j] = Nptr[M1ptr[i] * shape2 + j] +\
                                     u0ptr[M1ptr[shapeM1 + i] * shape2 + j]\
                            *(0.82*(2*Qptr[i] + Qptr[shapeM1 + i]) \
                                *M3ptr[M1ptr[4 * shapeM1 + i]*shape2 + j] + \
                                dt * 0.54*Qptr[i] * M4ptr[M1ptr[4 * shapeM1 + i] * shape2 + j])



    cdef complex128_t[:,::1] M5 = np.fft.fft(N)
        
    cdef complex128_t* M5ptr = &M5[0,0]
    cdef double* wptr = &w_tiled[0,0]


    for i in range(shape1):
        for j in range(shape2):
            M5ptr[i * shape2 + j] = wptr[i * shape2 + j] * M5ptr[i * shape2 + j]
    M5 =  np.fft.ifft(M5)

    M5ptr = &M5[0,0]

    for i in range(shape1):
        for j in range(shape2):
            Nptr[i * shape2 + j] = gam_no_aeff * ( Nptr[i * shape2 + j]  + tsh * M5ptr[i * shape2 + j])
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
    return N


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
    return N
