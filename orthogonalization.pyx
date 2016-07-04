

import numpy as np
cimport numpy as np
import params
# cimport params
from scipy import linalg

def calculate_matrix(np.ndarray mpp, np.ndarray mpm, np.ndarray mmp, np.ndarray mmm):
    cdef np.ndarray[np.float64_t, ndim = 2] result
    m1 = np.concatenate((mpp, mpm), axis=1)
    m2 = np.concatenate((mmp, mmm), axis=1)
    result = np.concatenate((m1, m2), axis = 0)
    return result

def calculate_H(np.ndarray S, np.ndarray A, np.ndarray R, double d, np.ndarray ni, int r4=0):
    cdef  np.ndarray[np.float64_t, ndim = 2] H, E, EPP, EPM, E_temp,dd
    EPP = np.zeros((ni.shape[0], ni.shape[0]))
    if r4 == 0:
        E_temp = np.tile(ni- 0.5 *d *d  , (ni.shape[0], 1))
    else:
        E_temp = np.tile((ni + 0.5)- 0.5 *d *d  , (ni.shape[0], 1))
    # print "E_temp"
    # print E_temp
    cdef Py_ssize_t i, j
    for i in range(ni.shape[0]):
        for j in range(ni.shape[0]):
            if i <j:
                EPP[i,j] = E_temp[i,j]
            if i==j:
                EPP[i,j]=E_temp[i,j] * 0.5

    EPP = EPP + np.transpose(EPP)
    EPM = E_temp
    # print "EPP"
    # print EPP
    EMP = np.transpose(EPP)
    e1 = np.concatenate((EPP, EPM), axis=1)
    e2 = np.concatenate((EPM.T, EPP), axis=1)
    E = np.concatenate((e1, e2), axis = 0)

    # print ("E (d=" + str(d) +"):", E)
    # dd = -0.5 * d * d * np.ones((S.shape[0], S.shape[1]))
    H =  A +R+ np.multiply(E , S)
    return H,E



