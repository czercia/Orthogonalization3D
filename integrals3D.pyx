from scipy import integrate
from scipy import special
import numpy as np
cimport numpy as np
import warnings
import params

def numerov_results(np.ndarray l, np.ndarray energy):
    cdef np.ndarray[object, ndim=1] result_x = np.empty((len(l)), dtype = np.object)
    cdef np.ndarray[object, ndim=1] result_y = np.empty((len(l)), dtype = np.object)
    # result_x = []
    # result_y = []
    cdef np.ndarray[np.float64_t, ndim = 2] numerov_res = np.zeros((1, 1))
    for ind in range(len(l)):
        numerov_res = np.loadtxt(params.directory_numerov + 'psi_l=' + str(l[ind]) + '_' + str(energy[ind]) + '.dat')
        numerov_res = numerov_res.T
        result_x[ind] = numerov_res[0]
        result_y[ind]= numerov_res[1]
    return result_x, result_y

def f(x, np.ndarray x_list, np.ndarray y_list):
    return np.interp(x, x_list, y_list)

def rmax_value(np.ndarray energy):
    cdef double r_max_value
    cdef np.ndarray[np.float64_t, ndim = 1] result = np.zeros((len(energy)), dtype=float)
    cdef Py_ssize_t i
    for i in range(len(energy)):
        if energy[i] <= -0.1:
            r_max_value = 8
        elif -0.1 < energy[i] < 0.1:
            r_max_value = 20
        elif 0.1 <= energy[i] < 0.3:
            r_max_value = 40
        elif 0.3 <= energy[i] < 0.5:
            r_max_value = 2.2 * np.sqrt(energy[i] / params.alpha)
        elif 0.5 <= energy[i] < 4:
            r_max_value = (2 - 0.1 * energy[i]) * np.sqrt(energy[i] / params.alpha)
        elif 4 <= energy[i] < 10:
            r_max_value = (2 - 0.05 * energy[i]) * np.sqrt(energy[i] / params.alpha)
        elif energy[i] >= 10:
            r_max_value = (2 - 0.01 * energy[i]) * np.sqrt(energy[i] / params.alpha) - 40
        result[i] = r_max_value
    return result

def rmax(int i, int j, np.ndarray r_max):
    return min(r_max[i], r_max[j])

def normalize(int i, double x_max,  np.ndarray numerov_x,
                      np.ndarray numerov_y):
    cdef double n
    n = integrate.quad(lambda x: abs(f(x, numerov_x[i], numerov_y [i])) * abs(f(x, numerov_x[i], numerov_y[i])), -x_max, x_max, epsabs=1e-6)[0]
    return 1. / np.sqrt(n)

def norm_matrix(int nst, np.ndarray r_max, np.ndarray numerov_x, np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    cdef np.ndarray[np.float64_t, ndim =1] norm_list = np.zeros((nst))
    for i in range(nst):
        norm_list[i] = normalize(i, r_max[i], numerov_x, numerov_y)
    for i in range(nst):
        for j in range(nst):
            result[i, j] = norm_list[i] * norm_list[j]
    print ('normalization calculated')
    return result

def norm_list(int nst, np.ndarray r_max, np.ndarray numerov_x, np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    cdef np.ndarray[np.float64_t, ndim =1] norm_li = np.zeros((nst))
    for i in range(nst):
        norm_li[i] = normalize(i, r_max[i], numerov_x, numerov_y)
    return norm_li


cdef double vp(double ro, double z,  double d, double b):
    cdef double res
    res = ((z + d) * (z + d) + ro * ro + b * b) * ((z + d) * (z + d) + ro * ro + b * b)
    return 1.0000 / res

cdef double vm(double x, double d, double b):
    cdef double res
    res = ((x - d) * (x - d) + b * b) * ((x - d) * (x - d) + b * b)
    return 1.0000 / res

def spp_1d_integrate(int nst, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            if i <= j:
                x_max = rmax(i, j, r_max)
                result[i, j] = \
                    integrate.quad(
                        lambda x: norm[i, j] * f(x, numerov_x[i], numerov_y[i]) * f(x, numerov_x[j], numerov_y[j]),
                        -x_max, x_max, epsabs=1e-6, limit=100)[
                        0]
            else:
                result[i, j] = result[j, i]
    return result

def spm_1d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            result[i, j] = \
                integrate.quad(
                    lambda x: norm[i, j] * f(x + d, numerov_x[i], numerov_y[i]) * f(x - d, numerov_x[j], numerov_y[j]),
                    -x_max + d, x_max - d, epsabs=1e-6,
                    limit=100)[
                    0]
    return result

def amm_1d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            if i <= j:
                x_max = rmax(i, j, r_max)
                result[i, j] = \
                    integrate.quad(
                        lambda x: norm[i, j] * f(x - d, numerov_x[i], numerov_y[i]) * (x * d) * f(x - d, numerov_x[j],
                                                                                                  numerov_y[j]), -x_max + d,
                        x_max +d ,  epsabs=1e-6, limit=100)[0]
            else:
                result[i, j] = result[j, i]
    return result
def app_1d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            if i <= j:
                x_max = rmax(i, j, r_max)
                result[i, j] = \
                    integrate.quad(
                        lambda x: norm[i, j] * f(x + d, numerov_x[i], numerov_y[i]) * (-x * d) * f(x + d, numerov_x[j],
                                                                                                   numerov_y[j]),
                        -x_max - d,
                        x_max -d ,  epsabs=1e-6, limit=100)[0]
            else:
                result[i, j] = result[j, i]
    return result
def apm_1d_integrate(int nst,  double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            result[i, j] = integrate.quad(
                lambda x: norm[i, j] * f(x + d, numerov_x[i], numerov_y[i]) * (x * d) * f(x - d, numerov_x[j],
                                                                                          numerov_y[j]), -x_max + d,
                x_max - d,  epsabs=1e-6, limit=100)[0]
    return result
# def amp_1d_integrate(int nst, np.ndarray ni, double x_max, double d, np.ndarray norm):
#     cdef Py_ssize_t i, j
#     cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
#     for i in range(nst):
#         for j in range(nst):
#             result[j, i] = integrate.quad(lambda x: norm[i, j] *f(x - d, ni[i]) * (-x * d) * f(x + d, ni[j]), -x_max,
#                                           x_max, epsabs=1e-6, limit=100)[0]
#     return result

def rpp_1d_integrate(int nst, double d, np.ndarray norm, double b, np.ndarray r_max,
                     np.ndarray numerov_x, np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            result[i, j] = \
                integrate.quad(
                    lambda x: norm[i, j] * f(x + d, numerov_x[i], numerov_y[i]) * (vp(x, d, b) - vm(x, d, b)) * f(x + d,
                                                                                                                  numerov_x[
                                                                                                                      j],
                                                                                                                  numerov_y[
                                                                                                                      j]),
                    -x_max -d,
                    x_max + d,  epsabs=1e-6, limit=100)[0]
    return result

def rmm_1d_integrate(int nst, double d, np.ndarray norm, double b, np.ndarray r_max,
                     np.ndarray numerov_x, np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            result[i, j] = \
                integrate.quad(
                    lambda x: norm[i, j] * f(x - d, numerov_x[i], numerov_y[i]) * (vm(x, d, b) - vp(x, d, b)) * f(x - d,
                                                                                                                  numerov_x[
                                                                                                                      j],
                                                                                                                  numerov_y[
                                                                                                                      j]),
                    -x_max + d,
                    x_max + d,  epsabs=1e-6, limit=100)[0]
    return result

def rpm_1d_integrate(int nst, double d, np.ndarray norm, double b, np.ndarray r_max,
                     np.ndarray numerov_x, np.ndarray numerov_y):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            result[i, j] = \
                integrate.quad(
                    lambda x: norm[i, j] * f(x + d, numerov_x[i], numerov_y[i]) * (vm(x, d, b) - vp(x, d, b)) * f(x - d,
                                                                                                                  numerov_x[
                                                                                                                      j],
                                                                                                                  numerov_y[
                                                                                                                      j]),
                    -x_max + d,
                    x_max - d , epsabs=1e-6, limit=100)[0]
    return result
# def rmp_1d_integrate(int nst, np.ndarray ni, double x_max, double d, np.ndarray norm):
#     cdef Py_ssize_t i, j
#     cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
#     for i in range(nst):
#         for j in range(nst):
#             result[j, i] = norm[i, j] * (f(2 * d, ni[j]) * f(0, ni[i]) - f(0, ni[j])* f(-2*d, ni[i]))
#     return result
