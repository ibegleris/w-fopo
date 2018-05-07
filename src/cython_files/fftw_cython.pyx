cdef extern from "fftw3.h":
    int fftw_init_threads()
    void fftw_plan_with_nthreads(int)
   
    cdef int FFTW_FORWARD
    cdef unsigned FFTW_ESTIMATE     

    ctypedef double fftw_complex[2]

    void *fftw_malloc(size_t)
    void fftw_free(void *)

    ctypedef struct _fftw_plan:
       pass

    ctypedef _fftw_plan *fftw_plan

    void fftw_execute(fftw_plan)
    void fftw_destroy_plan(fftw_plan)
    fftw_plan fftw_plan_dft_1d(int, fftw_complex*, fftw_complex*, int, unsigned)
    void fftw_print_plan(fftw_plan)

def fft(double[:] input, double[:] output):
    fftw_init_threads()
    fftw_plan_with_nthreads(4)
    cdef int n = input.shape[0]
    cdef double *tmp = <double *> fftw_malloc(n * sizeof(double))
    cdef int i
    for i in range(n):
        tmp[i] = input[i]
    cdef fftw_plan plan =  fftw_plan_dft_1d(n / 2,
                                           <fftw_complex *>tmp,
                                           <fftw_complex *>tmp,
                                           FFTW_FORWARD,
                                           FFTW_ESTIMATE)
    fftw_print_plan(plan)
    fftw_execute(plan)
    for i in range(n):
        output[i] = tmp[i]
    fftw_destroy_plan(plan)
    fftw_free(tmp)