

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

def calculate_H(np.ndarray S, np.ndarray A, np.ndarray R, double d, np.ndarray ni):
    cdef  np.ndarray[np.float64_t, ndim = 2] H, E, EPP, EPM, E_temp,dd
    EPP = np.zeros((ni.shape[0], ni.shape[0]))
    E_temp = np.tile(ni + 0.5, (ni.shape[0], 1))   #trzeba odpowiednie ni lub od razu E miec z bazy

    cdef Py_ssize_t i, j
    for i in range(ni.shape[0]):
        for j in range(ni.shape[0]):
            EPP[i,j] = E_temp[i,j] - params.alpha * d * d

    EPP = EPP + np.transpose(EPP)
    # EPM = E_temp
    EPM =  0.5 * (E_temp + np.transpose(E_temp)) - 2 * params.alpha * d * d * np.ones((E_temp.shape[0], E_temp.shape[0]))

    e1 = np.concatenate((EPP, EPM), axis=1)
    e2 = np.concatenate((EPM.T, EPP), axis=1)
    E = np.concatenate((e1, e2), axis = 0)

    H =  A + R + np.multiply(E , S)
    return H,E




