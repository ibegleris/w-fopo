#!/usr/bin/env python
# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
cimport numpy as cnp


from libc.string cimport memcpy

cdef extern from "Python.h":
    ctypedef int size_t

    void* PyMem_Malloc(size_t n)
    void PyMem_Free(void* buf)


# These are commented out in the numpy support we cimported above.
# Here I have declared them as taking void* instead of PyArrayDescr
# and object. In this file, only NULL is passed to these parameters.


cdef extern from "src/mklfft.h":
    int cdouble_cdouble_mkl_fft1d_out(cnp.ndarray, int, int, cnp.ndarray)
    int cdouble_cdouble_mkl_ifft1d_out(cnp.ndarray, int, int, cnp.ndarray)



# Initialize numpy
cnp.import_array()

ctypedef double complex complex128_t


cdef cnp.ndarray __allocate_result(cnp.ndarray x_arr, int f_type):
    """
    An internal utility to allocate an empty array for output of not-in-place FFT.
    """
    cdef cnp.npy_intp *f_shape
    cdef cnp.ndarray f_arr "ff_arrayObject"

    f_ndim = cnp.PyArray_NDIM(x_arr)

    f_shape = <cnp.npy_intp*> PyMem_Malloc(f_ndim * sizeof(cnp.npy_intp))
    memcpy(f_shape, cnp.PyArray_DIMS(x_arr), f_ndim * sizeof(cnp.npy_intp))

    # allocating output buffer
    f_arr = <cnp.ndarray> cnp.PyArray_EMPTY(
        f_ndim, f_shape, <cnp.NPY_TYPES> f_type, 0) # 0 for C-contiguous
    PyMem_Free(f_shape);

    return f_arr


# this routine implements complex forward/backward FFT
# Float/double inputs are not cast to complex, but are effectively
# treated as complexes with zero imaginary parts.
# All other types are cast to complex double.
cdef _fft1d_impl(complex128_t[:,::1] x_arr):
    """
    Uses MKL to perform 1D FFT on the input array x along the given axis.
    """
    cdef cnp.ndarray x = np.asarray(x_arr)
    
    cdef int shape = x_arr.shape[1]
    cdef cnp.ndarray f_arr =  __allocate_result(x, cnp.NPY_CDOUBLE);

    #cdouble_cdouble_mkl_ifft1d_out(x_arr, shape, 1, f_arr)
    cdouble_cdouble_mkl_fft1d_out(x, shape, 1, f_arr)

    return f_arr




def testing():
    from time import time
    a = np.random.randn(2,2**17) + 1j * np.random.randn(2,2**17)


    #print(output)
    N = 100
    tsp, tnp = [],[]
    for i in range(N):
        t1 = time()
        np.fft.fft(a)
        tnp.append(time() - t1) 
        t1 = time()
        _fft1d_impl(a)
        tsp.append(time() - t1)

    print('Numpy is {}'.format(np.average(tnp)))

    print('Spliced is {}'.format(np.average(tsp)))
    print('Speedup is at: {}'.format(np.average(tnp)/np.average(tsp) ))
    output = _fft1d_impl(a)
    np.testing.assert_allclose(output, np.fft.fft(a))
