from scipy import integrate
from scipy import special
import numpy as np
cimport numpy as np
import warnings
import params
cdef extern from "math.h":
    double sqrt(double x) nogil

# def numerov_results(np.ndarray l, np.ndarray energy):
#     cdef np.ndarray[object, ndim=1] result_x = np.empty((len(l)), dtype=np.object)
#     cdef np.ndarray[object, ndim=1] result_y = np.empty((len(l)), dtype=np.object)
#     cdef np.ndarray[np.float64_t, ndim=1] result_xmax = np.empty((len(l)), dtype=np.float)
#     cdef np.ndarray[np.float64_t, ndim = 2] numerov_res = np.zeros((1, 1))
#     for ind in range(len(l)):
#         numerov_res = np.loadtxt(params.directory_numerov + 'psi_l=' + str(l[ind]) + '_' + str(energy[ind]) + '.dat')
#         numerov_res = numerov_res.T
#         result_x[ind] = numerov_res[0]
#         result_y[ind] = numerov_res[1]
#         result_xmax = result_x[ind][-1]
#     return result_x, result_y, result_xmax

# def f(double ro, double z, double d, np.ndarray x_list, np.ndarray y_list):
#     cdef double r = sqrt(ro * ro + (z+d) * (z+d))
#     return np.interp(r, x_list, y_list)

def f(double ro, double z, double d):
    return 1

def f_sph(double r, np.ndarray x_list, np.ndarray y_list):
    return np.interp(r, x_list, y_list)

def psi(double r):
    return 1

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

def normalize(int i, double x_max):
    cdef double n
    n = 4 * np.pi * integrate.quad(lambda r: r * r * abs( psi(r)) * abs(psi(r)),
                       0,
                       x_max, epsabs=1e-6)[0]
    return 1. / np.sqrt(n)

def norm_matrix(int nst, np.ndarray r_max):
    cdef Py_ssize_t i, j
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    cdef np.ndarray[np.float64_t, ndim =1] norm_list = np.zeros((nst))
    for i in range(nst):
        norm_list[i] = normalize(i, r_max[i])
    for i in range(nst):
        for j in range(nst):
            result[i, j] = norm_list[i] * norm_list[j]
    print ('normalization calculated')
    return result

def norm_list(int nst, np.ndarray r_max):
    cdef Py_ssize_t i, j
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    cdef np.ndarray[np.float64_t, ndim =1] norm_li = np.zeros((nst))
    for i in range(nst):
        norm_li[i] = normalize(i, r_max[i])
    return norm_li

# cdef double vp(double ro, double z, double d, double b):
#     cdef double res
#     res = ((z + d) * (z + d) + ro * ro + b * b) * ((z + d) * (z + d) + ro * ro + b * b)
#     return 1.0000 / res
#
# cdef double vm(double ro, double z, double d, double b):
#     cdef double res
#     res = ((z - d) * (z - d) + ro * ro + b * b) * ((z - d) * (z - d) + ro * ro + b * b)
#     return 1.0000 / res

# def ssph_harm():
#     cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((params.l_max + 1, params.l_max + 1))
#     cdef Py_ssize_t i, j
#     for i in range(params.l_max + 1):
#         for j in range(params.l_max + 1):
#             if i <= j:
#                 result[i, j] = 2 * np.pi * integrate.quad(
#                     lambda teta: special.sph_harm(n=i, m=0, theta=0, phi=teta) * special.sph_harm(n=j, m=0, theta=0,
#                                                                                                   phi=teta) * np.sin(
#                         teta), 0, 2 * np.pi)[0]
#             else:
#                 result[i, j] = result[j, i]
#     return result

def spp_3d_integrate(int nst, np.ndarray norm, np.ndarray r_max):
    cdef Py_ssize_t i, j
    cdef double x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            if i <= j:
                x_max = rmax(i, j, r_max)
                result[i, j] = \
                        integrate.quad(
                            lambda r: norm[i, j] *r * r* psi(r) * psi(r),
                            0, x_max, epsabs=1e-6, limit=100)[
                            0]
            else:
                result[i, j] = result[j, i]
    return result

cdef double r(double ro, double z):
    cdef double result
    result = sqrt(ro * ro + z * z)
    return result

# cdef double leg_pol_norm(int l_i, int l_j):
#     cdef double result
#     result = sqrt((2 * l_i + 1) * (2 * l_j + 1)) / (2 * np.pi)
#     return result
#
# cdef double legendre(int l_i, int l_j, double ro, double z_i, double z_j):
#     return special.eval_legendre(l_i, z_i / r(ro, z_i)) * special.eval_legendre(l_j, z_j / r(ro, z_j))

def spm_3d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y, np.ndarray l):
    cdef Py_ssize_t i, j
    cdef double res1, res2, lim_low_z, lim_up_z
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst), dtype=np.float64)
    for i in range(nst):
        for j in range(nst):
            lim_low_z = -r_max[j] + d
            lim_up_z = (r_max[i] * r_max[i] - r_max[j] * r_max[j]) / (4 * d)
            res1 = integrate.dblquad(
                lambda ro, z: norm[i, j] * f(ro, z, d) * f(ro, z, - d) * ro,
                lim_low_z, lim_up_z, lambda z: 0, lambda z: sqrt(r_max[j] * r_max[j] - (z - d) * (z - d)),
                epsabs=1e-6,
                limit=100)[0]

            lim_low_z = (r_max[i] * r_max[i] - r_max[j] * r_max[j]) / (4 * d)
            lim_up_z = r_max[i] - d
            res2 = integrate.dblquad(
                lambda ro, z: norm[i, j] * f(ro, z, d) * f(ro, z, - d)  * ro,
                lim_low_z, lim_up_z, lambda z: 0, lambda z: sqrt(r_max[i] * r_max[i] - (z + d) * (z + d)),
                epsabs=1e-6,
                limit=100)[0]
            result[i, j] = res1 + res2
    return result

# def amm_3d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
#                      np.ndarray numerov_y):
#     cdef Py_ssize_t i, j
#     cdef double x_max
#     cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
#     for i in range(nst):
#         for j in range(nst):
#             if i <= j:
#                 x_max = rmax(i, j, r_max)
#                 result[i, j] = \
#                     integrate.quad(
#                         lambda x: norm[i, j] * f(x - d, numerov_x[i], numerov_y[i]) * (x * d) * f(x - d, numerov_x[j],
#                                                                                                   numerov_y[j]),
#                         -x_max + d,
#                         x_max + d, epsabs=1e-6, limit=100)[0]
#             else:
#                 result[i, j] = result[j, i]
#     return result
# def app_3d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
#                      np.ndarray numerov_y):
#     cdef Py_ssize_t i, j
#     cdef double x_max
#     cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
#     for i in range(nst):
#         for j in range(nst):
#             if i <= j:
#                 x_max = rmax(i, j, r_max)
#                 result[i, j] = \
#                     integrate.quad(
#                         lambda x: norm[i, j] * f(x + d, numerov_x[i], numerov_y[i]) * (-x * d) * f(x + d, numerov_x[j],
#                                                                                                    numerov_y[j]),
#                         -x_max - d,
#                         x_max - d, epsabs=1e-6, limit=100)[0]
#             else:
#                 result[i, j] = result[j, i]
#     return result

def appmm_3d_integrate(int nst, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
          np.ndarray numerov_y, np.ndarray l):
    cdef Py_ssize_t i, j
    cdef double x_max, res1, lim_low_x, lim_up_x
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst), dtype=np.float64)
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            lim_low_x = -x_max
            lim_up_x = x_max
            result[i, j] = integrate.dblquad(
                lambda ro, x: norm[i, j] * f(r(ro, x), numerov_x[i], numerov_y[i]) * f(
                    r(ro, x), numerov_x[j], numerov_y[j]) * legendre(l[i], l[j], ro, x,
                                                                     x) * ro * x,
                lim_low_x, lim_up_x, lambda x: 0, lambda x: sqrt(x_max * x_max - x * x),
                epsabs=1e-6,
                limit=100)[0]
            result[i, j] = leg_pol_norm(l[i], l[j]) * result[i, j]
    return result

def apm_3d_integrate(int nst, double d, np.ndarray norm, np.ndarray r_max, np.ndarray numerov_x,
                     np.ndarray numerov_y, np.ndarray l):
    cdef Py_ssize_t i, j
    cdef double res1, res2, lim_low_z, lim_up_z
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst), dtype=np.float64)
    for i in range(nst):
        for j in range(nst):
            lim_low_z = -r_max[j] + d
            lim_up_z = (r_max[i] * r_max[i] - r_max[j] * r_max[j]) / (4 * d)
            res1 = integrate.dblquad(
                lambda ro, z: norm[i, j] * f(r(ro, z + d), numerov_x[i], numerov_y[i]) * f(
                    r(ro, z - d), numerov_x[j], numerov_y[j]) * legendre(l[i], l[j], ro, z + d,
                                                                         z - d) * ro * z,
                lim_low_z, lim_up_z, lambda z: 0, lambda z: sqrt(r_max[j] * r_max[j] - (z - d) * (z - d)),
                epsabs=1e-6,
                limit=100)[0]

            lim_low_z = (r_max[i] * r_max[i] - r_max[j] * r_max[j]) / (4 * d)
            lim_up_z = r_max[i] - d
            res2 = integrate.dblquad(
                lambda ro, z: norm[i, j] * f(r(ro, z + d), numerov_x[i], numerov_y[i]) * f(
                    r(ro, z - d), numerov_x[j], numerov_y[j]) * legendre(l[i], l[j], ro, z + d,
                                                                         z - d) * ro * z,
                lim_low_z, lim_up_z, lambda z: 0, lambda z: sqrt(r_max[i] * r_max[i] - (z + d) * (z + d)),
                epsabs=1e-6,
                limit=100)[0]
            result[i, j] = leg_pol_norm(l[i], l[j]) * (res1 + res2)
    return result

def rpp_3d_integrate(int nst, double d, np.ndarray norm, double b, np.ndarray r_max,
                     np.ndarray numerov_x, np.ndarray numerov_y, np.ndarray l):
    cdef Py_ssize_t i, j
    cdef double lim_low_z, lim_up_z, x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            lim_low_z = -x_max - d
            lim_up_z = x_max - d
            result[i, j] = \
                integrate.dblquad(
                    lambda ro, z: norm[i, j] * f(r(ro, z + d), numerov_x[i], numerov_y[i]) * vm(ro, z, d, b) * f(
                        r(ro, z + d),
                        numerov_x[
                            j],
                        numerov_y[
                            j]),
                    lim_low_z,
                    lim_up_z, lambda z: 0, lambda z: sqrt(x_max * x_max - (z + d) * (z + d)), epsabs=1e-6, limit=100)[0]
            result[i, j] = leg_pol_norm(l[i], l[j]) * result[i, j]
    return result

def rmm_3d_integrate(int nst, double d, np.ndarray norm, double b, np.ndarray r_max,
                     np.ndarray numerov_x, np.ndarray numerov_y, np.ndarray l):
    cdef Py_ssize_t i, j
    cdef double lim_low_z, lim_up_z, x_max
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst))
    for i in range(nst):
        for j in range(nst):
            x_max = rmax(i, j, r_max)
            lim_low_z = -x_max + d
            lim_up_z = x_max + d
            result[i, j] = \
                integrate.dblquad(
                    lambda ro, z: norm[i, j] * f(r(ro, z - d), numerov_x[i], numerov_y[i]) * vp(ro, z, d, b) * f(
                        r(ro, z - d),
                        numerov_x[
                            j],
                        numerov_y[
                            j]),
                    lim_low_z,
                    lim_up_z, lambda z: 0, lambda z: sqrt(x_max * x_max - (z - d) * (z - d)), epsabs=1e-6, limit=100)[0]
            result[i, j] = leg_pol_norm(l[i], l[j]) * result[i, j]
    return result

def rpm_3d_integrate(int nst, double d, np.ndarray norm, double b, np.ndarray r_max,
                     np.ndarray numerov_x, np.ndarray numerov_y, np.ndarray l):
    cdef Py_ssize_t i, j
    cdef double res1, res2, lim_low_z, lim_up_z
    cdef np.ndarray[np.float64_t, ndim = 2] result = np.zeros((nst, nst), dtype=np.float64)
    for i in range(nst):
        for j in range(nst):
            lim_low_z = -r_max[j] + d
            lim_up_z = (r_max[i] * r_max[i] - r_max[j] * r_max[j]) / (4 * d)
            res1 = integrate.dblquad(
                lambda ro, z: norm[i, j] * f(r(ro, z + d), numerov_x[i], numerov_y[i]) * f(
                    r(ro, z - d), numerov_x[j], numerov_y[j]) * legendre(l[i], l[j], ro, z + d,
                                                                         z - d) * ro * vp(ro, z, d, b),
                lim_low_z, lim_up_z, lambda z: 0, lambda z: sqrt(r_max[j] * r_max[j] - (z - d) * (z - d)),
                epsabs=1e-6,
                limit=100)[0]

            lim_low_z = (r_max[i] * r_max[i] - r_max[j] * r_max[j]) / (4 * d)
            lim_up_z = r_max[i] - d
            res2 = integrate.dblquad(
                lambda ro, z: norm[i, j] * f(r(ro, z + d), numerov_x[i], numerov_y[i]) * f(
                    r(ro, z - d), numerov_x[j], numerov_y[j]) * legendre(l[i], l[j], ro, z + d,
                                                                         z - d) * ro * vp(ro, z, d, b),
                lim_low_z, lim_up_z, lambda z: 0, lambda z: sqrt(r_max[i] * r_max[i] - (z + d) * (z + d)),
                epsabs=1e-6,
                limit=100)[0]
            result[i, j] = leg_pol_norm(l[i], l[j]) * (res1 + res2)
    return result
